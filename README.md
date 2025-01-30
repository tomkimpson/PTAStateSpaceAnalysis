# Argus

![Tests](https://github.com/tomkimpson/Argus/actions/workflows/run_test.yml/badge.svg)


[![codecov](https://codecov.io/gh/tomkimpson/Argus/graph/badge.svg?token=2PEOHCFV1K)](https://codecov.io/gh/tomkimpson/Argus)


Welcome to the `Argus` repo.

This is a research project for the [detection of nHz gravitational waves](https://arxiv.org/abs/2105.13270) with PTAs, using a [state-space representation](https://en.wikipedia.org/wiki/State-space_representation). 

It is an ongoing effort to open-source methods developed at the [University of Melbourne](https://github.com/UniMelb-NSGW) and [OzGrav](https://www.ozgrav.org). Published literature that makes use of these methods include:

* [Kimpson et al. 2024a](https://arxiv.org/abs/2409.14613)
* [Kimpson et al 2024b](https://arxiv.org/abs/2410.10087)
* [Kimpson et al 2025](https://arxiv.org/abs/2501.06990)





### Notes

[Instructions on installing pulsar tools](https://gist.github.com/tomkimpson/5a245f6f1a8fc3b9cb39258741f7b572)


Useful example analysis from NANOGrav https://github.com/nanograv/15yr_stochastic_analysis/blob/main/tutorials/parameter_est.ipynb

The 15 year dataset from NANOGrav is a good standard to test on. I find parsing the older mock data using modern versions of enterprise/linbstempo/PINT does not work well. See e.g. `test_loading_mock_data.ipynb`







