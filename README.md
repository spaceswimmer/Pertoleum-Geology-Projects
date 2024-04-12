# Petroleum Geology Projects

This repository contains a collection of projects related to petroleum geology, primarily focusing on well log analysis and different petrophysical tasks. 
The projects are implemented using python, sklearn, numpy, matplotlib, seaborn, pandas, scipy and various optimization methods.

## Projects:

1. **Prediction Geophysical Logs with ML:**
   - Description: Geophysical well surveys are a key source of data for predicting the oil and gas potential of subsurface formations and are often lacking in full due to significant economic costs. The task of supplementing missing well data is relevant in oil companies and can be addressed using machine learning methods. This project focuses on forecasting well logs using gradient boosting techniques, particularly XGBoost. It explores the application of machine learning algorithms to predict well log data.
   - Technologies: Python, XGBoost, Gradient Boosting, Machine Learning

2. **Petrophysical Problems:**
   - Description: This projects addresses various petrophysical challenges encountered in petroleum geology. It involves the application of mathematical and computational methods to analyze well data, estimate reservoir properties, and optimize production strategies.
   - Technologies: Python, pandas, matplotlib, seaborn, numpy, Optimization Methods - scipy library
  
   a. **Spectral Gamma-ray Logging:**
   - Description: This project focuses on the analysis of spectral gamma-ray logs, which are crucial for correlating information from boreholes. The method is widely used in core-sampled wells. Concentrations of radioactive elements - U, Th, K are determined through least squares analysis.
   - Technologies: pandas, numpy, matplotlib

   b. **Density Logging:**
      - Description: Density logging data contains substantial amounts of data, which can often be approximated to simplify processing and further calculations. This project develops an algorithm to approximate density logging data through iterative partitioning of the original dataset and minimizing the total variance.
      - Technologies: numpy, pandas, matplotlib
   
   c. **Well Correlation using Levenshtein Distances:**
      - Description: Well correlation is essential in the exploration and development of every oilfield. Traditionally done manually by experts, this project applies a well correlation algorithm using descriptions of rock formations and the Levenshtein distance method - the difference between sequences of characters.
      - Technologies: numpy, pandas, matplotlib, seaborn
   
   d. **X-ray Diffraction (XRD) Mineral Analysis:**
      - Description: XRD mineral analysis allows for a detailed study of the structure of rocks. This project employs various optimization methods to identify peaks in the XRD spectrum - the characteristic signatures of specific minerals.
      - Technologies: scipy, numpy, pandas, matplotlib
   
   e. **Reservoir Sample Tomography:**
      - Description: Reservoir sample tomography is considered the most accurate and functional petrophysical method for studying pore space. However, the results yield a large amount of data per sample, making processing cumbersome. This project utilizes pandas with grouping and bin container creation to effectively segment pore spaces by diameter.
      - Technologies: pandas, numpy, matplotlib
        
   f. **Capillary Curve Modeling with Optimization:**
   - Description: Capillary curves provide essential insights for petroleum engineers to understand the height of the oil-saturated and water-saturated parts of the reservoir, crucial for reservoir development. However, manual analysis of these curves consumes a significant amount of time. There exist approximation models that involve finding coefficients using optimization methods. This project involved approximating 42 capillary curves using various models and the least squares method.
   - Technologies: scipy, numpy, pandas, matplotlib

## Usage:

Each project directory contains detailed documentation and instructions on how to use the code. Feel free to explore, experiment, and contribute to further advancements in petroleum geology research and applications.

## Contributions:

Contributions to this repository are welcome! Whether it's bug fixes, feature enhancements, or new project ideas, your contributions help foster collaboration and innovation within the petroleum geology community.

## License:

This repository is licensed under the [MIT License](LICENSE), allowing for open collaboration and use of the codebase for both academic and commercial purposes.

---

*Disclaimer: This repository is intended for educational and research purposes only. Any real-world applications should be thoroughly validated and verified by domain experts.*
