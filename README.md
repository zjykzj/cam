<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

 <div align="center"><a title="" href="git@github.com:zjykzj/cam.git"><img align="center" src="./imgs/cam.png"></a></div>

<p align="center">
  «cam» implements Gradient-weighted Class Activation Mapping.
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

![](./imgs/cmp.jpg)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

Through CAM (Class Activation Mapping), I can effectively visualize the model's focus on images, helping me better apply
them to practical scenarios.

This warehouse has implemented Grad-CAM (*Weight the 2D activations by the average gradient*). At present, there are
many other variants besides CAM and Grad-CAM. You can view the usage
of [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
and [frgfm/torch-cam](https://github.com/frgfm/torch-cam)

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/jacobgil/pytorch-grad-cam/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj