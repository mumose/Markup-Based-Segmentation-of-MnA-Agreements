import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import yaml
import argparse

from transformers import MarkupLMProcessor
from transformers import MarkupLMForTokenClassification

import torch
import evaluate

import utils
import input_pipeline

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

# TODO: add tensorboard functionality
# TODO: add new script for running inference on test set and obtaining metrics
# TODO: create slurm file for training on HPC
# TODO: look into label alignment
# TODO: is there value in labeling other subwords in each word

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_train_loop(batch, model, optimizer, loss_fct, device, train_metric, label_list):
    # get the inputs;
    inputs = {k: v.to(device) for k, v in batch.items()}

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


def run_eval_loop(eval_dataloader, model, device, eval_metric):
    model.eval()
    for batch in tqdm(eval_dataloader):
        # get the inputs;
        inputs = {k: v.to(device) for k, v in batch.items()}

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

    # preprocess the train and test dataset
    train_data = utils.get_dataset(
        config["data"]["train_contract_dir"], id2label, label2id
    )

    test_data = utils.get_dataset(
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

    test_dataset = input_pipeline.MarkupLMDataset(
        data=test_data, processor=processor, max_length=config["model"]["max_length"]
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["model"]["test_batch_size"], shuffle=False
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
    early_stop_ct = 0
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
            )

        # run eval loop
        run_eval_loop(test_dataloader, model, device, eval_metric)

        # compute the metrics at the end of each epoch
        train_metrics = utils.compute_metrics(train_metric)
        eval_metrics = utils.compute_metrics(eval_metric)

        # save the state dict for the best run
        if eval_metrics["overall_f1"] > best_eval_score:
            model_savepath = config["model"]["model_savepath"].split(".")[0]
            model_savepath = (
                f"{model_savepath}_{epoch}_{eval_metrics['overall_f1']:0.3f}.pt"
            )
            torch.save(model.state_dict(), model_savepath)
        else:
            early_stop_ct += 1

        print(
            f"Epoch {epoch} Train Metrics: {train_metrics}"
            + f"\n\nEval Metrics: {eval_metrics}"
        )

        if early_stop_ct >= config["model"]["early_stop_ct"]:
            break
    return model


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

    model = main(config)
