# MXNet 中的异常处理

这个教程解释了MXNet中的异常处理支持，并提供了例子当在一个多线程上下文中怎样抛出并解决异常。尽管，这个例子是用python写的，他们也能过容易的扩张到其他绑定的语言中。

MXNet 异常能够从两个区域抛出：

* MXNet 主线程。例如，推理与推断式
* 大量线程：

	* 依赖引擎并行执行计算
	* 通过迭代器，在数据加载阶段，文本解析阶段等。

在第一种情况下，异常抛出能在主线程中解决。在第二种情况下，在一个大量的线程中抛出异常，被捕获并传递到主线程中，之后重新抛出。这个教程将给予更多的解释和例子关于如何为第二种案例解决异常。

## 先决条件

完成这个教程，我们需要：
 
 * MXNet 

## 迭代器的异常处理

接下来的例子显示了如何为迭代器处理异常。在这个例子中，我们以较少数量的标签填充数据和标签的文件，而不是样本数量。这应该抛出一个异常。

CSVIter 使用 Prefetcherlter 来加载和解析数据。PrefetcherIter 在后台产生一个生产者线程来预读数据而主线程作消费数据。当label没有发现具体的样本时，生产者线程将抛出一个异常。

这个异常被传递给主线程，当调用Next作为接下来： for batch in iter(data_train) ，重新抛出异常。

通常，在python中，异常能过通过调用Next和BeforeFirst来抛出。他们对应的reset（）和next() 方法。

```

import os
import mxnet as mx

cwd = os.getcwd()
data_path = os.path.join(cwd, "data.csv")
label_path = os.path.join(cwd, "label.csv")

with open(data_path, "w") as fout:
    for i in range(8):
        fout.write("1,2,3,4,5,6,7,8,9,10\n")

with open(label_path, "w") as fout:
    for i in range(7):
        fout.write("label"+str(i))

try:
    data_train = mx.io.CSVIter(data_csv=data_path, label_csv=label_path, data_shape=(1, 10),
                               batch_size=4)

    for batch in iter(data_train):
        print(data_train.getdata().asnumpy())
except mx.base.MXNetError as ex:
    print("Exception handled")
    print(ex)
   
```

###限制

当您的最后一次next()调用没有到达异常发生的数据集中的批次时，有一个竞争条件。取决于哪个线程赢得竞赛，在此情况下可以抛出异常，也可以不抛出。为了避免这种情况，如果您认为它可以抛出需要处理的异常，则应该尝试遍历整个数据集。

## 算子异常处理

接下来的例子展示了在命令模式中如何为算子进行异常处理。

对于算子案例，如果依赖引擎运行在 ThreadedEnginePool 或者 ThreadedEnginePerDevice 模式，将产生许多线程。最终的算子在产生的线程中的一个中执行。

如果一个算子在执行过程中抛出异常，此异常从依赖链中传播。

