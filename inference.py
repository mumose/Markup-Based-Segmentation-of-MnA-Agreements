import os
import yaml
import argparse
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from IPython.display import display

import torch
import evaluate
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import MarkupLMProcessor
from transformers import MarkupLMForTokenClassification
from transformers import set_seed

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

import utils
import input_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# set the seed for the hugggingface package
set_seed(42)


def compute_auprc(all_labels_bin, all_preds, metric_basename):
    '''Computes the AUPRC by averaging the AP across all classes

    Args:
        all_labels_bin: np.array. Binarized version of the labels vector
            for all batches
        all_preds: np.array. Prediction vector for all batches
        metric_basename: str. The basename used to save the metrics

    Returns:
        auprc: float. The average precision averaged over all classes
    '''
    # Compute macro-averaged precision and recall
    precision, recall, average_precision = dict(), dict(), dict()
    precision["macro"], recall["macro"], _ = precision_recall_curve(all_labels_bin.ravel(),
                                                                    all_preds.ravel())
    average_precision["macro"] = average_precision_score(all_labels_bin,
                                                         all_preds,
                                                         average="macro") # test out weighted

    # compute the average precision per class
    # skip the 0 index since it corresponds to the outside tag
    for i in range(1, len(label_list)):
        precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i],
                                                            all_preds[:, i])
        average_precision[i] = average_precision_score(all_labels_bin[:, i],
                                                       all_preds[:, i])

    # create the plot for average precision/AUPRC
    plt.figure()
    plt.step(recall['macro'],
             precision['macro'],
             where='post',
             label='Macro-average precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Macro-Averaged Precision-Recall Curve: AP={0:0.2f}'.format(average_precision["macro"]))
    plt.legend(loc="lower right")

    fig = plt.gcf()
    plt.tight_layout()

    fig.savefig(f'{metric_basename}_pr_curve.png')

    return average_precision["macro"]


def get_batch_pred(batch, model, device,
                   metric, label_list, config):
    '''Runs inference loop for one batch of input

    Args:
        batch: dict[torch.Tensor]. Dict containing input tensors
            required for forward pass of MarkupLM
        model: transformers.PreTrainedModel. fine-tuned MarkupLM model
        device: torch.device. Specifies whether GPU is available for computation
        metric: evaluate.seqeval. The metric used for computing the f1-score
        label_list: list. List of labels used to train the MarkupLM model
        config: dict. Contains user-provided params and args

    Returns:
        preds: np.array. Pred tensor for valid token ids for single batch
        refs: np.array. Label vector for valid token ids for single batch
    '''
    # get the inputs;
    inputs = {k: v.to(device) for k, v in batch.items()}

    # if ablation mode is set to true then
    # either mask the xpaths or shuffle them
    if config["ablation"]["run_ablation"]:
        inputs = utils.ablation(config, inputs)

    # forward pass
    outputs = model(**inputs)

    # get the logits and predicted classes
    logits = outputs.logits
    labels = batch["labels"]

    # get the str repr of the predictions and labels. Used for
    # computing the f1-score
    str_preds, str_refs = utils.convert_preds_to_labels(logits.argmax(dim=-1),
                                                        labels,
                                                        label_list,
                                                        device)

    metric.add_batch(predictions=str_preds, references=str_refs)

    # get the preds and labels corresponding to valid tokens
    # drop the special tokens
    preds = logits.detach().cpu().squeeze(0).numpy()
    refs = labels.detach().cpu().squeeze(0).numpy()

    valid_idx = np.where(refs != -100)[0]

    preds = preds[valid_idx]
    refs = refs[valid_idx]

    return preds, refs


def run_inference_loop(dataloader, model, device,
                       metric, label_list, config,
                       class_weights):
    '''Runs inference loop for entire dataset and saves metrics to disk

    Args:
        dataloader: torch.utils.data.DataLoader: iterator over Dataset object
        model: transformers.PreTrainedModel. fine-tuned MarkupLM model
        device: torch.device. Specifies whether GPU is available for computation
        metric: evaluate.seqeval. The metric used for computing the f1-score
        label_list: list. List of labels used to train the MarkupLM model
        config: dict. Contains user-provided params and args
        class_weights: np.array. Array of class weights where each
            term is computed as num_examples_class / total_examples

    Returns:
        None
    '''
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc='inference_loop'):
        preds, refs = get_batch_pred(batch, model, device,
                                     metric, label_list, config)
        all_preds.append(preds)
        all_labels.append(refs)

    # concat the preds and label for each batch into single tensor
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # binarize the labels
    all_labels_bin = np.eye(len(label_list))[all_labels]

    # compute precision, recall, f1-score
    test_metrics = utils.compute_metrics(test_metric)

    metric_basename = config['predict']['model_path']
    metric_basename = os.path.basename(metric_basename).rsplit('.')[0]
    metric_basename = os.path.join(config['model']['collateral_dir'],
                                   metric_basename)

    # compute the area under precision-recall curve
    auprc = compute_auprc(all_labels_bin, all_preds, metric_basename)
    test_metrics['auprc'] = auprc

    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_savepath = os.path.join(collateral_dir,
                                         f"{metric_basename}_test_metrics.csv")
    test_metrics_df.to_csv(test_metrics_savepath, index=False)

    display(test_metrics_df)

    return


def main(config):
    '''Main execution of script'''
    # get the  list of labels along with the label to id mapping and
    # reverse mapping
    label_list, id2label, label2id = utils.get_label_list(config)

    print("*" * 50)
    print('Prepared Label List. Preparing Training Data ')
    print("*" * 50)

    test_data = utils.get_dataset(config["data"]["test_contract_dir"],
                                  id2label,
                                  label2id,
                                  data_split='test',
                                  num_contracts=None)

    print("*" * 50)
    print(f'Using Large Model: {config["model"]["use_large_model"]}')
    print("*" * 50)

    # define the processor and model
    if config["model"]["use_large_model"]:
        processor = MarkupLMProcessor.from_pretrained(
            "microsoft/markuplm-large", only_label_first_subword=True
        )
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-large", id2label=id2label, label2id=label2id
        )

    else:
        processor = MarkupLMProcessor.from_pretrained(
            "microsoft/markuplm-base", only_label_first_subword=True
        )
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-base", id2label=id2label, label2id=label2id
        )

    processor.parse_html = False

    # reload the model
    model_load_path = config['predict']['model_path']
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    # convert the input dataset
    # to torch datasets. Create the dataloaders as well
    test_dataset = input_pipeline.MarkupLMDataset(
        data=test_data,
        processor=processor,
        max_length=config["model"]["max_length"]
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    model.to(device)  # move to GPU if available

    print("*" * 50)
    print("Running Prediction Loop")

    # get the class weights to compute weigted sum across classes
    class_value_counts, class_weights = utils.get_class_dist(config["data"]["test_contract_dir"],
                                                             id2label,
                                                             label2id,
                                                             mode='normal')

    # create contrainer for computing test metrics
    test_metric = evaluate.load("seqeval",
                                scheme="IOBES",
                                mode="strict",
                                experiment_id='test',
                                keep_in_memory=True)

    run_inference_loop(test_dataloader, model, device,
                       test_metric, label_list, config,
                       class_weights)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to config file specifiying user params",
    )

    args = parser.parse_args()

    print("*" * 50)
    print('Processing...')
    print("*" * 50)

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    main(config)