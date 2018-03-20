
## GraphWave

基于多线程和GPU加速的 graphwave 实现方案。

fork自[这里](https://github.com/snap-stanford/graphwave)，也可以参考origin 分支

另外参考了[这里](https://github.com/bkj/graphwave)的实现

## 安装和使用

* 先安装相关软件包

```
pip install -r requirements.txt
```

如果有问题，可以尝试使用sudo、或者安装到当前用户目录等方式进行安装

* 测试

```
chmod 777 run.sh
./run.sh
```

* 使用

例如：

```
python main.py --n-chunks 16 --n-jobs 16 --inpath ./_data/synthetic/3200.edgelist --outpath ./_results/synthetic/3200
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
