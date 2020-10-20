[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/lukaskln/Making-predictions-based-on-highly-corrupted-MRI-data">
    <img src="https://github.com/lukaskln/Making-predictions-based-on-highly-corrupted-MRI-data/blob/main/Graphics/OutliersPC.png" alt="Logo" width="600">
  </a>

  <h3 align="center">The Effect of Sparse Coding visualized in Application</h3>

  <p align="center">
    Area of Anomaly Detection and Machine Learning.
    <br />
    <a href="https://github.com/lukaskln/Making-predictions-based-on-highly-corrupted-MRI-data"><strong>Explore the Project »</strong></a>
    <br />
    <br />
    <a href="https://github.com/lukaskln/Making-predictions-based-on-highly-corrupted-MRI-data/issues">Report Bug</a>
  </p>
</p>

## About the Project

Predicting the age of patients based on low quality magnetic resonance imaging (MRI) data. Main focus was on outlier removal via dimension reduction with Support Vector Classifiers (SVC) with radial kernel plus a small gamma value and feature selection by regression- and LASSO-based methods. For the pre-processing, the column median was inserted for missing values, zero variance features dropped, and a quantile transformer used for scaling and standardizing. For estimation and pre-processing evaluation three types of models were used: Support Vector Regression (SVR), boosted regression trees and a Deep Neural Network (DNN). All estimation, outlier detection and feature selection models were evaluated via 5-fold cross validation and hyperparameter optimization by grid search.

## The Code 

The code is structured into three scripts. First the explorative analysis where empirical proof for feature selection, transformation and dropped outlier is given. Second two scripts with the full pipelines for the boosted regression tree plus DNN and the SVR. The pre-processing in both pipelines differs.

## Contact

Lukas Klein - [LinkedIn](https://www.linkedin.com/in/lukasklein1/) - lukas.klein@etu.unige.ch

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/lukaskln/Making-predictions-based-on-highly-corrupted-MRI-data/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/lukasklein1/
