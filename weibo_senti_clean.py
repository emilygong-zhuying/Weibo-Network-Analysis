import os
import paddle
import paddlenlp
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from functools import partial
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding

def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        next(f)
        for line in f:
            words = line[:-3]
            labels = line[-2]
            yield {'text': words, 'labels': labels}

train_ds = load_dataset(read, data_path='train_clean.txt',lazy=False)
dev_ds = load_dataset(read, data_path='dev_clean.txt',lazy=False)

model_name = "ernie-3.0-medium-zh"
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=len(train_ds.label_list))
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_example(example, tokenizer):
    tokenized_example = tokenizer(text=example['text'], max_seq_length=128)
    # 加上label用于训练
    tokenized_example['label'] = [int(example['labels'])]
    return tokenized_example

trans_func = partial(convert_example, tokenizer=tokenizer)

train_ds = train_ds.map(trans_func)
dev_ds = dev_ds.map(trans_func)

collate_fn = DataCollatorWithPadding(tokenizer)

# 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
train_batch_sampler = BatchSampler(train_ds, batch_size=32, shuffle=True)
dev_batch_sampler = BatchSampler(dev_ds, batch_size=64, shuffle=False)
train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

# Adam优化器、交叉熵损失函数、accuracy评价指标
optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for step, batch in enumerate(data_loader, start=1):
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

        # 计算模型输出、损失函数值、分类概率值、准确率
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()
    print("eval accu: %.5f" % (acc))
    model.train()
    return acc

# 开始训练
import time
import paddle.nn.functional as F

epochs = 5 # 训练轮次
ckpt_dir = "ernie_ckpt_cleaner" #训练过程中保存模型参数的文件夹
best_acc = 0
best_step = 0
global_step = 0 #迭代次数
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

        # 计算模型输出、损失函数值、分类概率值、准确率
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        # 每迭代10次，打印损失函数值、准确率、计算速度
        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # 每迭代100次，评估当前训练的模型、保存当前模型参数和分词器的词表等
        if global_step % 100 == 0:
            save_dir = ckpt_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(global_step, end=' ')
            print(time.time())
            acc_eval = evaluate(model, metric, dev_data_loader)
            if acc_eval > best_acc:
                best_acc = acc_eval
                best_step = global_step

                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)