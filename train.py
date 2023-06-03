import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os
import yaml
import argparse
import pandas as pd

from tqdm.auto import tqdm

import torch
import evaluate
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import MarkupLMProcessor
from transformers import MarkupLMForTokenClassification

import utils
import input_pipeline


# TODO: add tensorboard functionality
# TODO: add new script for running inference on eval set and obtaining metrics
# TODO: create slurm file for training on HPC
# TODO: look into label alignment
# TODO: is there value in labeling other subwords in each word

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_train_loop(
    batch, model, optimizer, loss_fct, device, train_metric, label_list, config
):
    # get the inputs;
    inputs = {k: v.to(device) for k, v in batch.items()}

    if config["ablation"]["run_ablation"]:
        inputs = utils.ablation(config, inputs)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(**inputs)

    # get the logits
    logits = outputs.logits

    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]

    # Only keep active parts of the loss
    num_labels = len(label_list)
    active_loss = attention_mask.view(-1) == 1
    active_logits = logits.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        labels.view(-1),
        torch.tensor(loss_fct.ignore_index).type_as(labels),
    )
    loss = loss_fct(active_logits, active_labels)

    # loss = outputs.loss
    loss.backward()
    optimizer.step()

    print("Train Loss:", loss.item())

    predictions = outputs.logits.argmax(dim=-1)
    labels = batch["labels"]
    preds, refs = utils.convert_preds_to_labels(predictions, labels, label_list, device)

    train_metric.add_batch(
        predictions=preds,
        references=refs,
    )

    return


def run_eval_loop(eval_dataloader, model, device, eval_metric, config):
    model.eval()
    for batch in tqdm(eval_dataloader):
        # get the inputs;
        inputs = {k: v.to(device) for k, v in batch.items()}

        if config["ablation"]["run_ablation"]:
            inputs = utils.ablation(config, inputs)

        # forward + backward + optimize
        outputs = model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        preds, refs = utils.get_labels(predictions, labels)

        eval_metric.add_batch(predictions=preds, references=refs)

    return


def main(config):
    # get the  list of labels along with the label to id mapping and
    # reverse mapping
    label_list, id2label, label2id = utils.get_label_list(config)

    # preprocess the train and eval dataset
    train_data = utils.get_dataset(
        config["data"]["train_contract_dir"], id2label, label2id
    )

    eval_data = utils.get_dataset(
        config["data"]["eval_contract_dir"], id2label, label2id
    )

    processor = MarkupLMProcessor.from_pretrained(
        "microsoft/markuplm-base", only_label_first_subword=False
    )
    processor.parse_html = False

    # convert the input dataset
    # to torch datasets. Create the dataloaders as well
    train_dataset = input_pipeline.MarkupLMDataset(
        data=train_data, processor=processor, max_length=config["model"]["max_length"]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["model"]["train_batch_size"], shuffle=True
    )

    eval_dataset = input_pipeline.MarkupLMDataset(
        data=eval_data, processor=processor, max_length=config["model"]["max_length"]
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config["model"]["eval_batch_size"], shuffle=False
    )

    # define the model
    if config["model"]["use_large_model"]:
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-large", id2label=id2label, label2id=label2id
        )
    else:
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-base", id2label=id2label, label2id=label2id
        )

    # get the class weights used to weigh the different terms in the loss fn
    class_weights = utils.get_class_dist(
        config["data"]["train_contract_dir"], id2label, label2id
    )

    # define the optimizer and loss fct
    optimizer = AdamW(model.parameters(), lr=config["model"]["learning_rate"])

    loss_fct = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights), ignore_index=config["model"]["ignore_index"]
    )

    # define the train and eval metric containers
    train_metric = evaluate.load("seqeval", scheme="BILOU", mode="strict")
    eval_metric = evaluate.load("seqeval", scheme="BILOU", mode="strict")

    model = model.to(device)  # move to GPU if available

    model.train()
    best_eval_score = -float("int")
    num_epochs_lower_eval = 0
    train_metrics_list, eval_metrics_list = [], []
    for epoch in range(config["model"]["num_epochs"]):
        model.train()
        for train_batch in tqdm(train_dataloader):
            run_train_loop(
                train_batch,
                model,
                optimizer,
                loss_fct,
                device,
                train_metric,
                label_list,
                config,
            )

        # run eval loop
        run_eval_loop(eval_dataloader, model, device, eval_metric, config)

        # compute the metrics at the end of each epoch
        train_metrics = utils.compute_metrics(train_metric)
        eval_metrics = utils.compute_metrics(eval_metric)

        train_metrics['epoch'] = epoch
        train_metrics_list.append(train_metrics)

        eval_metrics['epoch'] = epoch
        eval_metrics_list.append(eval_metrics)

        # save the state dict for the best run
        if eval_metrics["overall_f1"] > best_eval_score:
            model_savepath = config["model"]["model_savepath"].split(".")[0]
            model_savepath = (
                f"{model_savepath}_{epoch}_{eval_metrics['overall_f1']:0.3f}.pt"
            )
            torch.save(model.state_dict(), model_savepath)

            # reset the patience counter for early stopping
            num_epochs_lower_eval = 0
        else:
            num_epochs_lower_eval += 1
            print(f"Eval f1 score did not improve. Patience={num_epochs_lower_eval}")

        print(
            f"Epoch {epoch} Train Metrics: {train_metrics}"
            + f"\n\nEval Metrics: {eval_metrics}"
        )

        if num_epochs_lower_eval >= config["model"]["early_stop_patience"]:
            print("*" * 50)
            print("reached max patience limit. Stopping Training")
            print("*" * 50)
            break

    return model, train_metrics_list, eval_metrics_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to config file specifiying user params",
    )

    args = parser.parse_args()

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    model_dir = os.path.dirname(config['model']['model_savepath'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    metric_dir = config['model']['metric_dir']
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir, exist_ok=True)

    model, train_metrics_list, eval_metrics_list = main(config)

    train_metrics_df = pd.DataFrame(train_metrics_list)
    eval_metrics_df = pd.DataFrame(eval_metrics_list)

    train_metrics_savepath = os.path.join(metric_dir, "train_metrics.csv")
    train_metrics_df.to_csv(train_metrics_df, index=False)

    eval_metrics_savepath = os.path.join(metric_dir, "eval_metrics.csv")
    eval_metrics_df.to_csv(eval_metrics_df, index=False)