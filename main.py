""" Finetuning BERT/RoBERTa models on WinoGrande. """

from __future__ import absolute_import, division, print_function
from util import *


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

def train(args, train_dataset, dev_dataset, test_dataset, model, tokenizer, data_loader):
    """ Train the model """
    set_seed(args)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_pct is None:
        scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=math.floor(args.warmup_pct * t_total),
                                         num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0],
                              mininterval=1, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'question_ids': batch[0],
                      'question_mask': batch[1],
                      'cand1_ids': batch[2],
                      'cand1_mask': batch[3],
                      'cand2_ids': batch[4],
                      'cand2_mask': batch[5],
                      'knowledge_ids': batch[6],
                      'knowledge_mask': batch[7],
                      'cand1_path_ids': batch[8],
                      'cand1_path_mask': batch[9],
                      'cand2_path_ids': batch[10],
                      'cand2_path_mask': batch[11],
                      'topological_path_ids': batch[12],
                      'topological_path_mask': batch[13],
                      'labels': batch[14]}
            outputs = model(**inputs)
            loss = outputs[0]  

            if args.n_gpu > 1:
                loss = loss.mean()  
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  
                        evaluate_by_types(args, data_loader, tokenizer, 'data/test.json', model)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model 
                    model_to_save.encoder.save_pretrained(output_dir)
                    torch.save(model_to_save.state_dict(), output_dir + '/state_dict')
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


    return global_step, tr_loss / global_step



def evaluate_by_types(args, data_loader, tokenizer, data_file, model):
    HardPCR_data = data_loader.get_examples(data_file, 'HardPCR', args.helpful_only)
    HardPCR_features = convert_examples_to_features(args, HardPCR_data, args.max_seq_length, tokenizer)
    HardPCR_set = data_loader.get_dataset(HardPCR_features)
    HardPCR_correct, HardPCR_total = evaluate(args, HardPCR_set, model, 'HardPCR')
    print('HarPCR number:', HardPCR_total)
    CommonsenseQA_data = data_loader.get_examples(data_file, 'CommonsenseQA', args.helpful_only)
    CommonsenseQA_features = convert_examples_to_features(args, CommonsenseQA_data, args.max_seq_length, tokenizer)
    CommonsenseQA_set = data_loader.get_dataset(CommonsenseQA_features)
    CommonsenseQA_correct, CommonsenseQA_total = evaluate(args, CommonsenseQA_set, model, 'CommonsenseQA')
    print('CommonsenseQA number:', CommonsenseQA_total)
    COPA_data = data_loader.get_examples(data_file, 'COPA', args.helpful_only)
    COPA_features = convert_examples_to_features(args, COPA_data, args.max_seq_length, tokenizer)
    COPA_set = data_loader.get_dataset(COPA_features)
    COPA_correct, COPA_total = evaluate(args, COPA_set, model, 'COPA')
    print('COPA number:', COPA_total)
    ATOMIC_data = data_loader.get_examples(data_file, 'ATOMIC', args.helpful_only)
    ATOMIC_features = convert_examples_to_features(args, ATOMIC_data, args.max_seq_length, tokenizer)
    ATOMIC_set = data_loader.get_dataset(ATOMIC_features)
    ATOMIC_correct, ATOMIC_total = evaluate(args, ATOMIC_set, model, 'ATOMIC')
    print('ATOMIC number:', ATOMIC_total)

    all_correct = HardPCR_correct+CommonsenseQA_correct+COPA_correct+ATOMIC_correct
    all_total = HardPCR_total+CommonsenseQA_total+COPA_total+ATOMIC_total

    print('overall accuracy:', all_correct, '/', all_total, all_correct/all_total)



def evaluate(args, data, model, eval_name='All'):
    eval_dataset = data


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    score_list = []
    results = dict()
    for batch in tqdm(eval_dataloader, desc="Evaluating "+eval_name, mininterval=1, ncols=100):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'question_ids': batch[0],
                      'question_mask': batch[1],
                      'cand1_ids': batch[2],
                      'cand1_mask': batch[3],
                      'cand2_ids': batch[4],
                      'cand2_mask': batch[5],
                      'knowledge_ids': batch[6],
                      'knowledge_mask': batch[7],
                      'cand1_path_ids': batch[8],
                      'cand1_path_mask': batch[9],
                      'cand2_path_ids': batch[10],
                      'cand2_path_mask': batch[11],
                      'topological_path_ids': batch[12],
                      'topological_path_mask': batch[13],
                      'labels': batch[14]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
            pair_ids = batch[6].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pair_ids = np.append(pair_ids, batch[6].detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)

    tmp_correctness = 0
    for i in range(len(preds)):
        if preds[i] == out_label_ids[i]:
            tmp_correctness += 1
        else:
            tmp_correctness += 0
    results[eval_name + '_accuracy'] = tmp_correctness / len(preds)
    print(eval_name+' Accuracy:', tmp_correctness / len(preds))

    return tmp_correctness, len(preds)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut names")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_type", default='All', type=str,
                        help="Training data type, out of ('All', 'HardPCR', 'CommonsenseQA', 'COPA',  'ATOMIC')")
    parser.add_argument("--test_type", default='All', type=str,
                        help="Test data type")
    parser.add_argument("--helpful_only", action='store_true',
                        help="whether only select the helpful cases or not.")
    parser.add_argument("--use_knowledge", action='store_true',
                        help="whether to use knowledge or not.")
    parser.add_argument("--no_question", action='store_true',
                        help="whether to use question or not.")
    parser.add_argument("--train_number", default=100000, type=int,
                        help="The maximum training number")
    parser.add_argument("--model", default='baseline', type=str,
                        help="What model you want to test")
    parser.add_argument("--num_walk", default='5', type=int,
                        help="number of random walks")
    parser.add_argument("--walk_length", default='5', type=int,
                        help="Length of the random walk")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=80, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_prediction", action='store_true',
                        help="Whether to run prediction on the test set. (Training will not be executed.)")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--run_on_test', action='store_true')

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=10000, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_pct", default=0.1, type=float,
                        help="Linear warmup over warmup_pct*total_steps.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=True)
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task="winogrande")

    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large']:
        encoder_model = model_class.from_pretrained(args.model_name_or_path,
                                                    from_tf=bool('.ckpt' in args.model_name_or_path),
                                                    config=config)

        model = K2G(config, encoder_model, args)
    else:
        encoder_model = model_class.from_pretrained(args.model_name_or_path,
                                                    from_tf=bool('.ckpt' in args.model_name_or_path),
                                                    config=config)

        model = K2G(config, encoder_model, args)
        model.load_state_dict(torch.load(args.model_name_or_path + '/state_dict'))

    if args.local_rank == 0:
        torch.distributed.barrier()  
    model.to(args.device)
    data_loader = CKBQADataLoader(args, 'data', tokenizer)

    if args.local_rank == 0:
        torch.distributed.barrier()  
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_prediction:
        results = {}
        logger.info("Prediction on the test set (note: Training will not be executed.) ")
        evaluate(args, data_loader.test_set, model)
        evaluate_by_types(args, data_loader, tokenizer, 'data/test.json', model)
        logger.info("***** Experiment finished *****")

    if args.do_train:
        global_step, tr_loss = train(args, data_loader.train_set, data_loader.dev_set, data_loader.test_set, model, tokenizer, data_loader)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  
        model_to_save.encoder.save_pretrained(args.output_dir)
        torch.save(model_to_save.state_dict(), args.output_dir + '/state_dict')
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        encoder_model = model_class.from_pretrained(args.output_dir,
                                                    from_tf=bool('.ckpt' in args.model_name_or_path),
                                                    config=config)

        model = JointI(config, encoder_model, args)
        model.load_state_dict(torch.load(args.output_dir + '/state_dict'))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    results = {}
    checkpoints = [args.output_dir]
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(args.output_dir + '/' + 'pytorch_model.bin', recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN) 
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, data_loader.dev_set, model)
            evaluate_by_types(args, data_loader, tokenizer, 'data/test.json', model)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    logger.info("***** Experiment finished *****")
    return results


if __name__ == "__main__":
    main()
