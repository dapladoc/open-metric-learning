from copy import deepcopy
from pprint import pprint
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from oml.const import (
    BLUE,
    EMBEDDINGS_KEY,
    GRAY,
    GREEN,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    LOG_TOPK_IMAGES_PER_ROW,
    LOG_TOPK_ROWS_PER_METRIC,
    N_GT_SHOW_EMBEDDING_METRICS,
    OVERALL_CATEGORIES_KEY,
    PATHS_KEY,
    RED,
    X1_KEY,
    X2_KEY,
    Y1_KEY,
    Y2_KEY,
)
from oml.ddp.utils import is_main_process
from oml.functional.metrics import (
    TMetricsDict,
    _to_tensor,
    apply_mask_to_ignore,
    calc_gt_mask,
    calc_mask_to_ignore,
    calc_retrieval_metrics,
    calc_topological_metrics,
    reduce_metrics,
)
from oml.interfaces.metrics import IMetricDDP, IMetricVisualisable
from oml.interfaces.post_processor import IPostprocessor
from oml.metrics.accumulation import Accumulator
from oml.utils.images.images import get_img_with_bbox, square_pad
from oml.utils.misc import flatten_dict

TMetricsDict_ByLabels = Dict[Union[str, int], TMetricsDict]


class EmbeddingMetrics(IMetricVisualisable):

    metric_name = ""

    def __init__(
        self,
        model: torch.nn.Module,
        embeddings_key: str = EMBEDDINGS_KEY,
        labels_key: str = LABELS_KEY,
        is_query_key: str = IS_QUERY_KEY,
        is_gallery_key: str = IS_GALLERY_KEY,
        extra_keys: Tuple[str, ...] = (),
        cmc_top_k: Tuple[int, ...] = (5,),
        precision_top_k: Tuple[int, ...] = (5,),
        map_top_k: Tuple[int, ...] = (5,),
        fmr_vals: Tuple[int, ...] = tuple(),
        pfc_variance: Tuple[float, ...] = (0.5,),
        categories_key: Optional[str] = None,
        postprocessor: Optional[IPostprocessor] = None,
        metrics_to_exclude_from_visualization: Iterable[str] = (),
        check_dataset_validity: bool = True,
        return_only_main_category: bool = False,
        visualize_only_main_category: bool = True,
        verbose: bool = True,
        bs: bool = 128,
    ):
        self.model = model
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.extra_keys = extra_keys
        self.cmc_top_k = cmc_top_k
        self.precision_top_k = precision_top_k
        self.map_top_k = map_top_k
        self.fmr_vals = fmr_vals
        self.pfc_variance = pfc_variance

        self.categories_key = categories_key
        self.postprocessor = postprocessor

        self.distance_matrix = None
        self.mask_gt = None
        self.metrics = None
        self.mask_to_ignore = None
        self.bs = bs

        self.check_dataset_validity = check_dataset_validity
        self.visualize_only_main_category = visualize_only_main_category
        self.return_only_main_category = return_only_main_category

        self.metrics_to_exclude_from_visualization = ["fnmr@fmr", "pcf", *metrics_to_exclude_from_visualization]
        self.verbose = verbose

        self.keys_to_accumulate = [self.embeddings_key, self.is_query_key, self.is_gallery_key, self.labels_key]
        if self.categories_key:
            self.keys_to_accumulate.append(self.categories_key)
        if self.extra_keys:
            self.keys_to_accumulate.extend(list(extra_keys))

        self.acc = Accumulator(keys_to_accumulate=self.keys_to_accumulate)

    def setup(self, num_samples: int) -> None:  # type: ignore
        self.distance_matrix = None
        self.mask_gt = None
        self.metrics = None
        self.mask_to_ignore = None

        self.acc.refresh(num_samples=num_samples)

    def update_data(self, data_dict: Dict[str, Any]) -> None:  # type: ignore
        self.acc.update_data(data_dict=data_dict)

    def _calc_matrices(self) -> None:
        embeddings = self.acc.storage[self.embeddings_key]
        labels = self.acc.storage[self.labels_key]
        is_query = self.acc.storage[self.is_query_key]
        is_gallery = self.acc.storage[self.is_gallery_key]

        if self.postprocessor:
            # we have no this functionality yet
            self.postprocessor.process()

        # Note, in some of the datasets part of the samples may appear in both query & gallery.
        # Here we handle this case to avoid picking an item itself as the nearest neighbour for itself
        self.mask_to_ignore = calc_mask_to_ignore(is_query=is_query, is_gallery=is_gallery)
        self.mask_gt = calc_gt_mask(labels=labels, is_query=is_query, is_gallery=is_gallery)
        self.distance_matrix = calc_distance_matrix(
            model=self.model, embeddings=embeddings, is_query=is_query, is_gallery=is_gallery, bs=self.bs
        )

    def compute_metrics(self) -> TMetricsDict_ByLabels:  # type: ignore
        if not self.acc.is_storage_full():
            raise ValueError(
                f"Metrics have to be calculated on fully collected data. "
                f"The size of the current storage is less than num samples: "
                f"we've collected {self.acc.collected_samples} out of {self.acc.num_samples}."
            )

        self._calc_matrices()

        args_retrieval_metrics = {
            "cmc_top_k": self.cmc_top_k,
            "precision_top_k": self.precision_top_k,
            "map_top_k": self.map_top_k,
            "fmr_vals": self.fmr_vals,
        }
        args_topological_metrics = {"pfc_variance": self.pfc_variance}

        metrics: TMetricsDict_ByLabels = dict()

        # note, here we do micro averaging
        metrics[self.overall_categories_key] = calc_retrieval_metrics(
            distances=self.distance_matrix,
            mask_gt=self.mask_gt,
            mask_to_ignore=self.mask_to_ignore,
            check_dataset_validity=self.check_dataset_validity,
            reduce=False,
            **args_retrieval_metrics,  # type: ignore
        )

        embeddings = self.acc.storage[self.embeddings_key]
        metrics[self.overall_categories_key].update(calc_topological_metrics(embeddings, **args_topological_metrics))

        if self.categories_key is not None:
            categories = np.array(self.acc.storage[self.categories_key])
            is_query = self.acc.storage[self.is_query_key]
            query_categories = categories[is_query]

            for category in np.unique(query_categories):
                mask = query_categories == category

                metrics[category] = calc_retrieval_metrics(
                    distances=self.distance_matrix[mask],  # type: ignore
                    mask_gt=self.mask_gt[mask],  # type: ignore
                    mask_to_ignore=self.mask_to_ignore[mask],  # type: ignore
                    reduce=False,
                    **args_retrieval_metrics,  # type: ignore
                )

                mask = categories == category
                metrics[category].update(calc_topological_metrics(embeddings[mask], **args_topological_metrics))

        self.metrics_unreduced = metrics
        self.metrics = reduce_metrics(metrics)  # type: ignore

        if self.return_only_main_category:
            metric_to_return = {
                self.overall_categories_key: deepcopy(self.metrics[self.overall_categories_key])  # type: ignore
            }
        else:
            metric_to_return = deepcopy(self.metrics)

        if self.verbose and is_main_process():
            print("\nMetrics:")
            pprint(metric_to_return)

        return metric_to_return  # type: ignore

    def visualize(self) -> Tuple[Collection[plt.Figure], Collection[str]]:
        """
        Visualize worst queries by metrics.
        """
        metrics_flat = flatten_dict(self.metrics, ignored_keys=self.metrics_to_exclude_from_visualization)
        figures = []
        titles = []
        for metric_name in metrics_flat:
            if self.visualize_only_main_category and not metric_name.startswith(OVERALL_CATEGORIES_KEY):
                continue
            fig = self.get_plot_for_worst_queries(
                metric_name=metric_name, n_queries=LOG_TOPK_ROWS_PER_METRIC, n_instances=LOG_TOPK_IMAGES_PER_ROW
            )
            log_str = f"top {LOG_TOPK_ROWS_PER_METRIC} worst by {metric_name}".replace("/", "_")
            figures.append(fig)
            titles.append(log_str)
        return figures, titles

    def ready_to_visualize(self) -> bool:
        return PATHS_KEY in self.extra_keys

    def get_worst_queries_ids(self, metric_name: str, n_queries: int) -> List[int]:
        metric_values = flatten_dict(self.metrics_unreduced)[metric_name]  # type: ignore
        return torch.topk(metric_values, min(n_queries, len(metric_values)), largest=False)[1].tolist()

    def get_plot_for_worst_queries(
        self, metric_name: str, n_queries: int, n_instances: int, verbose: bool = False
    ) -> plt.Figure:
        query_ids = self.get_worst_queries_ids(metric_name=metric_name, n_queries=n_queries)
        return self.get_plot_for_queries(query_ids=query_ids, n_instances=n_instances, verbose=verbose)

    def get_plot_for_queries(self, query_ids: List[int], n_instances: int, verbose: bool = True) -> plt.Figure:
        """
        Visualize the predictions for the query with the indicies <query_ids>.

        Args:
            query_ids: Index of the query
            n_instances: Amount of the predictions to show
            verbose: wether to show image paths or not

        """
        assert self.metrics is not None, "We are not ready to plot, because metrics were not calculated yet."

        dist_matrix_with_inf, mask_gt = apply_mask_to_ignore(self.distance_matrix, self.mask_gt, self.mask_to_ignore)

        is_query = self.acc.storage[self.is_query_key]
        is_gallery = self.acc.storage[self.is_gallery_key]

        query_paths = np.array(self.acc.storage[PATHS_KEY])[is_query]
        gallery_paths = np.array(self.acc.storage[PATHS_KEY])[is_gallery]

        if all([k in self.acc.storage for k in [X1_KEY, X2_KEY, Y1_KEY, Y2_KEY]]):
            bboxes = list(
                zip(
                    self.acc.storage[X1_KEY],
                    self.acc.storage[Y1_KEY],
                    self.acc.storage[X2_KEY],
                    self.acc.storage[Y2_KEY],
                )
            )
        elif all([k not in self.acc.storage for k in [X1_KEY, X2_KEY, Y1_KEY, Y2_KEY]]):
            fake_coord = np.array([float("nan")] * len(is_query))
            bboxes = list(zip(fake_coord, fake_coord, fake_coord, fake_coord))
        else:
            raise KeyError(f"Not all the keys collected in storage! {[*self.acc.storage]}")

        query_bboxes = torch.tensor(bboxes)[is_query]
        gallery_bboxes = torch.tensor(bboxes)[is_gallery]

        fig = plt.figure(figsize=(30, 30 / (n_instances + N_GT_SHOW_EMBEDDING_METRICS + 1) * len(query_ids)))
        for j, query_idx in enumerate(query_ids):
            ids = torch.argsort(dist_matrix_with_inf[query_idx])[:n_instances]

            n_gt = mask_gt[query_idx].sum()  # type: ignore

            plt.subplot(
                len(query_ids),
                n_instances + 1 + N_GT_SHOW_EMBEDDING_METRICS,
                j * (n_instances + 1 + N_GT_SHOW_EMBEDDING_METRICS) + 1,
            )

            img = get_img_with_bbox(query_paths[query_idx], query_bboxes[query_idx], BLUE)
            img = square_pad(img)
            if verbose:
                print("Q  ", query_paths[query_idx])
            plt.imshow(img)
            plt.title(f"Query, #gt = {n_gt}")
            plt.axis("off")

            for i, idx in enumerate(ids):
                color = GREEN if mask_gt[query_idx, idx] else RED  # type: ignore
                if verbose:
                    print("G", i, gallery_paths[idx])
                plt.subplot(
                    len(query_ids),
                    n_instances + N_GT_SHOW_EMBEDDING_METRICS + 1,
                    j * (n_instances + 1 + N_GT_SHOW_EMBEDDING_METRICS) + i + 2,
                )
                img = get_img_with_bbox(gallery_paths[idx], gallery_bboxes[idx], color)
                img = square_pad(img)
                plt.title(f"{i} - {round(dist_matrix_with_inf[query_idx, idx].item(), 3)}")
                plt.imshow(img)
                plt.axis("off")

            gt_ids = mask_gt[query_idx].nonzero(as_tuple=True)[0][:N_GT_SHOW_EMBEDDING_METRICS]  # type: ignore

            for i, gt_idx in enumerate(gt_ids):
                plt.subplot(
                    len(query_ids),
                    n_instances + N_GT_SHOW_EMBEDDING_METRICS + 1,
                    j * (n_instances + 1 + N_GT_SHOW_EMBEDDING_METRICS) + i + n_instances + 2,
                )
                img = get_img_with_bbox(gallery_paths[gt_idx], gallery_bboxes[gt_idx], GRAY)
                img = square_pad(img)
                plt.title("GT " + str(round(dist_matrix_with_inf[query_idx, gt_idx].item(), 3)))
                plt.imshow(img)
                plt.axis("off")

        fig.tight_layout()
        return fig


class EmbeddingMetricsDDP(EmbeddingMetrics, IMetricDDP):
    def sync(self) -> None:
        self.acc = self.acc.sync()


def calc_distance_matrix(
    model: torch.nn.Module,
    embeddings: Union[np.ndarray, torch.Tensor],
    is_query: Union[np.ndarray, torch.Tensor],
    is_gallery: Union[np.ndarray, torch.Tensor],
    bs: int,
) -> torch.Tensor:
    assert all(isinstance(vector, (np.ndarray, torch.Tensor)) for vector in [embeddings, is_query, is_gallery])
    assert is_query.ndim == 1 and is_gallery.ndim == 1 and embeddings.ndim == 2
    assert embeddings.shape[0] == len(is_query) == len(is_gallery)

    embeddings, is_query, is_gallery = map(_to_tensor, [embeddings, is_query, is_gallery])

    query_mask = is_query == 1
    gallery_mask = is_gallery == 1
    query_embeddings = embeddings[query_mask]
    gallery_embeddings = embeddings[gallery_mask]

    distance_matrix = pairwise_dist(model=model, x1=query_embeddings, x2=gallery_embeddings, bs=bs)

    return distance_matrix


def pairwise_dist(model: torch.nn.Module, x1: torch.Tensor, x2: torch.Tensor, bs: int) -> torch.Tensor:
    original_device = x1.device
    x1 = x1.to(next(model.parameters()))
    x2 = x2.to(next(model.parameters()))
    i1 = torch.arange(x1.shape[0])
    i2 = torch.arange(x2.shape[0])
    i1, i2 = torch.meshgrid(i1, i2)
    i1 = i1.reshape(-1)
    i2 = i2.reshape(-1)
    i = 0
    scores = torch.empty(i1.shape[0], device=original_device)
    mode = model.training
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(total=i1.shape[0])
        while i * bs < i1.shape[0]:
            _a = i * bs
            _b = min((i + 1) * bs, i1.shape[0])
            _i1 = i1[_a:_b]
            _i2 = i2[_a:_b]
            _x1 = x1[_i1, :]
            _x2 = x2[_i2, :]
            scores[_a:_b] = model.forward(_x1, _x2).to(original_device).squeeze()
            i += 1
            progress_bar.update(_b - _a)
        progress_bar.close()
    if mode:
        model.train()
    scores = scores.reshape(x1.shape[0], x2.shape[0])
    return scores
