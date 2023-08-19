<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="git@github.com:zjykzj/cam.git"><img align="center" src="./imgs/cam.svg"></a></div>

<p align="center">
  «cam»实现了梯度加权类激活图
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

* [Grad-CAM](https://blog.zhujian.life/posts/531cdffe.html)

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [使用](#使用)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

通过CAM（类激活映射），我可以有效地可视化模型对于图像的关注区域，以此来帮助我更好地将它们应用于实际场景。

该仓库已实现Grad-CAM（*按平均梯度对2D激活进行加权*）。目前，除了CAM和Grad-CAM之外，还有许多其他变体。可以查看[jacobgil/pytorch grad-cam](https://github.com/jacobgil/pytorch-grad-cam)和[frgf/torch cam](https://github.com/frgfm/torch-cam)

## 使用

```shell
# Custom Grad-CAM
python imagenet/Grad-CAM.py --arch resnet50 imgs/cat.jpg
```

![](./imgs/cmp.jpg)

```shell
# jacobgil/pytorch-grad-cam
python samples/pytorch-grad-cam.py imgs/cat.jpg
```

![](./imgs/cmp_v2.jpg)

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

* [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjykzj/cam/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2023 zjykzj