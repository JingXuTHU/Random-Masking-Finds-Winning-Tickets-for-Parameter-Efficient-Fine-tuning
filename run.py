import os
import logging
import argparse
import time
import tasks
import random
import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, \
    TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
from metrics import calculate_metric
from utils import *
from PEFT import *

os.environ['WANDB_MODE'] = 'disabled'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2"  # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 0  # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None  # (only enabled with training) number of development samples
    num_eval: int = None  # number of evaluation samples
    num_train_sets: int = None  # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None  # designated seed to sample training samples/demos
    result_file: str = None  # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m"  # HuggingFace model name
    max_length: int = 2048  # max length the model can take
    auto_device: bool = True  # turn this on for zero2 and off for zero3

    # Training
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = False  # take the log likelihood of all options and train as classification

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Evaluation
    eval_batch_size: int = 8  # batch size for evaluation

    # Saving
    save_model: bool = False  # whether to save the model
    tag: str = ""  # saving tag

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # save model when interrupted (useful for long training)

    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    prefix_init_by_real_act: bool = True  # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA

    # Adapter
    adapter: bool = False  # use adapter
    adapter_act_type: str = 'relu'  # activation function for adapter
    adapter_r: int = 8  # r in adapter

    # AdaLora
    adalora: bool = False  # use AdaLora

    # Random Masking
    random_masking: bool = False  # use random masking
    masking_prob: float = 0.0  # masking probability for random masking (also for structured masking)

    # Structrued Masking
    structured_masking: bool = False  # use structured masking

    # Bitfit
    bitfit: bool = False  # use bitfit


def parse_args():
    # parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Configure other GPUs to suppress all log output
    if args.local_rank > 0:
        logger.setLevel(level=logging.CRITICAL)

    logger.info(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer, self.config = self.load_model()

    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model"):
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.auto_device:
                torch_dtype = torch.float16  # for OPT models, use float16; for llama models, use bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0
        if "llama" in self.args.model_name:
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))

        if self.args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.args.prefix_tuning:
            PrefixTuning(model, num_prefix=self.args.num_prefix, init_by_real_act=True)
        elif self.args.lora:
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha)
        elif self.args.adalora:
            # load_best_model_at_end in the run script should be turned off
            from peft import AdaLoraModel, AdaLoraConfig, TaskType
            adalora_config = AdaLoraConfig(peft_type="ADALORA", task_type=TaskType.CAUSAL_LM,
                                           r=self.args.lora_r, lora_alpha=self.args.lora_alpha,
                                           target_modules=["q_proj", "v_proj"])
            model = AdaLoraModel(model, adalora_config, "default")
        elif self.args.adapter:
            Adapter(model, r=self.args.adapter_r)
        elif self.args.random_masking:
            true_masking_prob = convert_masking_prob(self.args.model_name, self.args.masking_prob)
            logger.info(f"true masking prob: {true_masking_prob}")
            RandomMasking(model, masking_prob=true_masking_prob)
        elif self.args.structured_masking:
            true_masking_prob = convert_masking_prob(self.args.model_name, self.args.masking_prob)
            logger.info(f"true masking prob: {true_masking_prob}")
            StructuredMasking(model, masking_prob=true_masking_prob)
        elif self.args.bitfit:
            Bitfit(model)

        return model, tokenizer, config

    def forward(self, input_ids, attention_masks=None, option_len=None, generation=False, batch_size=None,
                num_of_candidates_arr=None):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor(input_ids).to(self.model.device)
        attention_masks = torch.tensor(attention_masks).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, attention_mask=attention_masks, do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0],
                              self.tokenizer.eos_token_id],
            )

            # For generation, directly return the text output
            output_texts = []
            for idx in range(len(outputs)):
                output_text = self.tokenizer.decode(outputs[idx][input_ids[idx].size(0):],
                                                    skip_special_tokens=True).strip()
                output_texts.append(output_text)
            return output_texts
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids, attention_mask=attention_masks).logits

            old_input_ids = input_ids
            old_logits = logits
            old_option_len = option_len
            input_ids = []
            logits = []
            option_len = []
            idx = 0
            for i in range(batch_size):
                input_ids.append(old_input_ids[idx:idx + num_of_candidates_arr[i]])
                logits.append(old_logits[idx:idx + num_of_candidates_arr[i]])
                option_len.append(old_option_len[idx:idx + num_of_candidates_arr[i]])
                idx += num_of_candidates_arr[i]

            selected_log_probs = []
            for idx1 in range(batch_size):
                tmp = []
                for idx2 in range(num_of_candidates_arr[idx1]):
                    padding_len = 0
                    label = input_ids[idx1][idx2][1 + padding_len:]
                    logit = logits[idx1][idx2][padding_len:-1]
                    log_probs = F.log_softmax(logit, dim=-1)

                    selected_log_prob = log_probs[torch.arange(len(label)).to(label.device), label]
                    selected_log_prob = selected_log_prob.cpu().detach()
                    tmp.append(selected_log_prob[-option_len[idx1][idx2]:])
                selected_log_probs.append(tmp)

            return selected_log_probs

    def one_step_pred(self, eval_samples):
        """
        Return the prediction on the eval sample.
        """
        batch_size = len(eval_samples)

        encoded_candidates, attention_masks, option_lens = encode_prompt_eval(
            self.task, self.task.get_template(), eval_samples, self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        predictions = []
        if self.task.generation:
            output_texts = self.forward(encoded_candidates, attention_masks=attention_masks, generation=True,
                                        batch_size=batch_size)
            for idx in range(len(eval_samples)):
                predictions.append(Prediction(correct_candidate=eval_samples[idx].correct_candidate,
                                              predicted_candidate=output_texts[idx]))
        else:
            num_of_candidates_arr = [len(eval_samples[i].candidates) for i in range(batch_size)]
            selected_log_probs = self.forward(encoded_candidates, attention_masks=attention_masks,
                                              option_len=option_lens, batch_size=batch_size,
                                              num_of_candidates_arr=num_of_candidates_arr)

            scores = [[x.mean().item() for x in outputs] for outputs in selected_log_probs]

            for idx in range(len(eval_samples)):
                if isinstance(eval_samples[idx].correct_candidate, list):
                    # For some datasets there are multiple correct answers
                    correct_candidate_id = [eval_samples[idx].candidates.index(c) for c in
                                            eval_samples[idx].correct_candidate]
                else:
                    correct_candidate_id = eval_samples[idx].candidates.index(eval_samples[idx].correct_candidate)

                predictions.append(
                    Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores[idx]))))

        return predictions

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        self.model.eval()
        torch.cuda.empty_cache()

        # Prediction loop
        predictions = []
        batched_eval_samples = []
        eval_batch_size = self.args.eval_batch_size
        for idx in range(len(eval_samples) // eval_batch_size):
            batched_eval_samples.append(eval_samples[idx * eval_batch_size:(idx + 1) * eval_batch_size])
        if len(eval_samples) % eval_batch_size != 0:
            batched_eval_samples.append(eval_samples[-(len(eval_samples) % eval_batch_size):])

        assert (one_train_set_per_eval_sample is False and train_samples == [])
        for batched_eval_id, batched_eval_sample in enumerate(tqdm(batched_eval_samples)):
            with torch.no_grad():
                predictions.extend(self.one_step_pred(batched_eval_sample))

        # Calculate metrics
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt_train(
                    self.task, self.task.get_template(), [], sample, self.tokenizer,
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True,
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id,
                                  "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in
                                 range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                 "labels": encoded_candidates[correct_candidate_id],
                                 "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                 "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))

        if self.args.only_train_option:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        collator = DataCollatorForTokenClassification

        from transformers import Trainer

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
                                                            pad_to_multiple_of=8) if self.args.train_as_classification else collator(
                self.tokenizer, pad_to_multiple_of=8),
        )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()

        # FSDP compatibility
        self.model = trainer.model

        # Reset the forward function for evaluation
        if self.args.only_train_option:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    """
    Get the result file tag
    """
    if not os.path.exists("result/"):
        os.makedirs("result/")

    save_model_name = args.model_name.split("/")[-1]
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + customized_tag


def main():
    args = parse_args()
    set_seed(args.seed)

    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval,
                                        num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    for train_set_id, train_samples in enumerate(train_sets):
        train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

        # Sample eval samples
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        # Prepare train and dev samples
        if args.num_dev is not None:
            dev_samples = train_samples[-args.num_dev:]
            train_samples = train_samples[:-args.num_dev]
        else:
            dev_samples = None

        logger.info(f"Train set {train_set_id} has {len(train_samples)} training samples, "
                    f"{len(dev_samples)} dev samples, and {len(eval_samples)} eval samples")

        # Training
        framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

        # Evaluation
        metrics = framework.evaluate([], eval_samples)

        logger.info("===== Train set %d =====" % train_set_seed)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(
                args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)


if __name__ == "__main__":
    main()
