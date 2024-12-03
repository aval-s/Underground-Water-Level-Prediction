# Underground-Water-Level-Prediction
Chapter 1
Introduction


To deal with the challenges related to weather resilience and sustainability principles, the importance of urban groundwater should be included into city making plan and design. Groundwater structures are dynamic and adjust usually to short-term and long-term changes in weather, groundwater withdrawal, and land use. Water level measurements from commentary wells are predominant supply of records about the hydrologic stresses appearing on the aquifers and how these stresses affect ground-water recharge, storage, and discharge. In this studies we focus on underground water level of India.

Traditionally groundwater levels are modelled with process-based models, which rely on the profound knowledge of the observed system dynamics. They require many additional spatial data on geological and hydrological properties of the aquifer. On the other hand, in data-driven modelling with machine-learning techniques our model is based solely on the data and some domain-specific knowledge is incorporated in to the system via appropriate data transformation (within engineering of new attributes). The goal in such a scenario would be to predict groundwater levels based on temporal data inputs (historic groundwater and surface water level data) and outputs (groundwater level). The model captures underlying processes based on the data without additional expert user input. In our work we present the whole data-mining pipeline, including exploratory data analysis, data pre-processing and modelling, where we explore accuracy and other benefits of a variety of modelling techniques on the same dataset (from interpretable modelling techniques such as Proceedings 2018, 2, 697 2 of 8 multivariate linear regression, linear SVM and decision trees to black-box models such as gradient boosted trees, random forests and artificial neural networks), which has rarely been reported in scientific literature for water-related scenarios. Claimed already a decade ago that data-driven modelling has overcome the initial stage and the main objectives shifted from method development and testing to the construction of useful architectures and applications of data-driven modelling for decision makers, according to the availability of the data. In reality, however, machine learning is still seeking its way into the practice and many studies have been published recently, which are researching the usability of different machine learning algorithms. Time-series techniques (ARX, ARMAX, ARMA, ARIMA) and artificial neural network models have been tested in random forests and maximum entropy have been tested in focused on multilayer perceptron networks. With R2 scores higher than 0.8 they claim that data-driven approach can be used as an alternative to process modelling techniques. Machine learning models have been used to study geological-geomechanically properties of districts in India.
 

Chapter 2
Problem Statement and Project Objectives


Problem : In many areas, groundwater is often one of the major sources of water supply for domestic, urban, agricultural and industrial purposes, especially in arid and semi-arid areas. However, many problems occur due to overexploitation of groundwater and unsustainable groundwater use and management, such as major water-level declines, drying up of wells, water-quality degradation, increased pumping costs, land surface subsidence, loss of pump-age in residential water supply wells, and aquifer compaction. These problems are becoming a serious issue globally, especially in developing countries. To secure water for the future, the sustainable management of groundwater resources in conjunction with surface water has urgently become the need of the hour. Accurate and reliable prediction of groundwater levels is a crucial component for achieving this goal, especially in watersheds in arid and semi-arid regions that are more susceptible to hydrological extreme events in the form of droughts. Also, groundwater is heavily used both for irrigation and industrialization in India. Due to faulty irrigation system, a lot of groundwater is wasted. Prediction of groundwater levels is the need of the hour to avoid future crisis.

OBJECTIVE : The main objectives are as follows:
	The purpose is to develop prediction model of highly accurate groundwater level forecasting that can be used to help water managers, engineers, and stake-holders manage groundwater in a more effective and sustainable manner.
	The goal in such a scenario would be to predict groundwater levels based on temporal data inputs (historic groundwater level data).
	The main idea behind this work is to study time-series techniques like SARIMA model and ARIMA are utilized for forecasting groundwater levels, including exploratory data analysis, data pre-processing and modeling, where we explore accuracy and other benefits.

 

Chapter 3
Technical Design & Methodology

3.1 Background Study
With the rapid development Internet of Things (IoT) in recent years, there has been an explosion in time series data. Based on the growth trends of database types in DB-Engines over the past two years, the growth of time series database has been immense. Implementations of these large open source time series databases are different, and none of them are perfect. However, the advantages of these databases can be combined to implement a perfect time series database.
3.1.1 Time series
A time series is a sequential set of data points, measured typically over successive times. It is mathematically defined as a set of vectors x(t),t = 0,1,2,... where t represents the time elapsed. The variable x(t) is treated as a random variable. The measurements taken during an event in a time series are arranged in a proper chronological order. A time series containing records of a single variable is termed as univariate. But if records of more than one variable are considered, it is termed as multivariate. A time series can be continuous or discrete. In a continuous time series observations are measured at every instance of time, whereas a discrete time series contains observations measured at discrete points of time. For example temperature readings, flow of a river, concentration of a chemical process etc. can be recorded as a continuous time series. On the other hand population of a particular city, production of a company, exchange rates between two different currencies may represent discrete time series. Usually in a discrete time series the consecutive observations are recorded at equally spaced time intervals such as hourly, daily, weekly, monthly or yearly time separations. The variable being observed in a discrete time series is assumed to be measured as a continuous variable using the real number scale. Furthermore a continuous time series can be easily transformed to a discrete one by merging data together over a specified time interval.

3.1.2 Components of Time Series
A time series in general is supposed to be affected by four main components, which can be separated from the observed data. These components are: Trend, Cyclical, Seasonal and Irregular components. The general tendency of a time series to increase, decrease or stagnate over a long period of time is termed as Secular Trend or simply Trend. Thus, it can be said that trend is a long term movement in a time series. For example, series relating to population growth, number of houses in a city etc. show upward trend, whereas downward trend can be observed in series relating to mortality rates, epidemics, etc. Seasonal variations in a time series are fluctuations within a year during the season. The important factors causing seasonal variations are: climate and weather conditions, customs, traditional habits, etc. For example sales of ice-cream increase in summer, sales of woollen cloths increase in winter. Seasonal variation is an important factor for businessmen, shopkeeper and producers for making proper future plans.

The cyclical variation in a time series describes the medium-term changes in the series, caused by circumstances, which repeat in cycles. The duration of a cycle extends over longer period of time, usually two or more years. Most of the economic and financial time series show some kind of cyclical variation. For example a business cycle consists of four phases, viz. i) Prosperity, ii) Decline, iii) Depression and iv) Recovery. Schematically a typical business cycle can be shown as below:








Fig. 2.1: A four phase business cycle

Irregular or random variations in a time series are caused by unpredictable influences, which are not regular and also do not repeat in a particular pattern. These variations are caused by incidences such as war, strike, earthquake, flood, revolution, etc. There is no defined statistical technique for measuring random fluctuations in a time series.

Considering the effects of these four components, two different types of models are generally used for a time series viz. Multiplicative and Additive models.

Multiplicative Model:  Y(t) = T(t)× S(t)×C(t)× I(t )
Additive Model: Y(t) = T(t) + S(t) + C(t) + I(t)
Here  Y(t is the observation and ) T(t), S(t), C(t) and I(t) are respectively the trend, seasonal, cyclical and irregular variation at time .t Multiplicative model is based on the assumption that the four components of a time series are not necessarily independent and they can affect one another; whereas in the additive model it is assumed that the four components are independent of each other.

3.1.3 How time series data different from other data?
Time series data have a natural temporal ordering. This makes time series analysis distinct from other common data analysis techniques, in which there is no natural ordering of the observations.

3.1.4 Time Series Notation
A number of different notations are in use for time-series analysis. A common notation specifying a time series X that is indexed by the natural numbers is written X = {X1, X2, X3, X4, X5...}

3.1.5 Time Series Models
A Time series model (probability model) will generally reflect the fact that observations close together in time will be more closely related than observations further apart. In addition, time series models will often make use of the natural one-way ordering of time so that values for a given period will be expressed as deriving in some way from past values, rather than from future values. A time series model for the observed data {Xt} is a specification of the joint distribution (or possibly only the means and covariances) of sequences of random variables {Xt} of which {Xt} is postulated to be a realization. Models for time series data can have many forms and represent different stochastic processes. When modelling variations in the level of a process, three broad classes of practical importance are the autoregressive (AR) models, the integrated (I) models, and the moving average (MA) models. These three classes depend linearly on previous data points. Combinations of these ideas produce autoregressive-moving average (ARMA) and autoregressive integrated moving average (ARIMA) models. Also, time series data which has seasonality (SARIMA), also known as seasonal ARIMA.

3.1.6 Time series Analysis
In practice a suitable model is fitted to a given time series and the corresponding parameters are estimated using the known data values. The procedure of fitting a time series to a proper model is termed as Time Series Analysis. It comprises methods that attempt to understand the nature of the series and is often useful for future forecasting and simulation. In time series forecasting, past observations are collected and analysed to develop a suitable mathematical model which captures the underlying data generating process for the series. The future events are then predicted using the model. This approach is particularly useful when there is not much knowledge about the statistical pattern followed by the successive observations or when there is a lack of a satisfactory explanatory model. Time series forecasting has important applications in various fields. Often valuable strategic decisions and precautionary measures are taken based on the forecast results. Thus making a good forecast, i.e. fitting an adequate model to a time series is very important. Over the past several decades many efforts have been made by researchers for the development and improvement of suitable time series forecasting models. 

3.1.7 General Approach to Time Series Modelling
• Plot the series and examine the main feature whether there is
	Trend
	Seasonal component
	Any apparent sharp changes in behaviour
	Any outlying observations

• Remove the trend and seasonal components to get stationary residuals by applying a Preliminary transformation to the data. For example, if the magnitude of the fluctuations appears to grow roughly linearly with the level of the series, then the transformed series {ln X1,...,ln Xn} will have fluctuations of more constant magnitude.. (If some of the data are negative, add a positive constant to each of the data values to ensure that all values are positive before taking logarithms.) Other ways by estimating the components and subtracting them from the data, and others depending on differencing the data, i.e., replacing the original series {Xt} by{Yt:= Xt − Xt −d} for some positive integer d .

• Choose a model to fit the residuals, making use of various sample statistics including the sample autocorrelation function

• Forecasting will be achieved by forecasting the residuals and then inverting the transformations described above to arrive at forecasts of the original series {Xt}.Other approach is transform series in its Fourier components (residual waves of different frequencies).This is important in signal processing and structural design. Now we get fully formed statistical models for stochastic simulation purposes, so as to generate alternative versions of the time series, representing what might happen over non- specific time-periods in the future

3.1.8 Akaike Information Criterion (AIC)
Akaike information criterion is a measure of relative goodness of fit of a statistical time series model. It based on the concept of information entropy, in effect offering a relative measure of the information lost when a given model is used to describe reality. AIC values provide a means for model selection. It can tell nothing about how well a model fits a data in an absolute sense. If all the candidate models fit poorly, AIC will not give any warning of that.
AIC  2k  2log(L)
Where k=number of parameters, L= maximized value of likelihood function for the estimated model. Given a set of candidate models for the data, the preferred model is the one with the minimum AIC value. Hence AIC not only rewards goodness of fit, but also includes a penalty that is an increasing function of the number of estimated parameters

3.2 Technical Design Architecture
In general models for time series data can have many forms and represent different stochastic processes. There are two widely used linear time series models in literature, viz. Autoregressive (AR) and Moving Average (MA) models. Combining these two, the Autoregressive Moving Average (ARMA) and Autoregressive Integrated Moving Average (ARIMA) [6, 21, 23] models have been proposed in literature. The Autoregressive Fractionally Integrated Moving Average (ARFIMA) model generalizes ARMA and ARIMA models. For seasonal time series forecasting, a variation of ARIMA, viz. the Seasonal Autoregressive Integrated Moving Average (SARIMA) model is used. 

3.2.1 The Autoregressive Moving Average (ARMA) Models
An ARMA(p, q) model is a combination of AR(p) and MA(q) models and is suitable for univariate time series modelling. In an AR(p) model the future value of a variable is assumed to be a linear combination of p past observations and a random error together with a constant term. Mathematically the AR(p)  model can be expressed as: 
y_t=c+∑_(j=1)^p▒〖φ_i y_(t-1)+ε_t=c+ φ_i y_(t-1)+φ_i y_(t-2)+⋯+φ_i y_(t-p)+ε_t 〗
Here yt and εt are respectively the actual value and random error (or random shock) at time period t , ϕi (i =1,2,..., p) are model parameters and c is a constant. The integer constant p is known as the order of the model. Sometimes the constant term is omitted for simplicity. 
Usually For estimating parameters of an AR process using the given time series, the YuleWalker equations are used.  
Just as an AR(p)  model regress against past values of the series, an MA(q) model uses past errors as the  explanatory variables. The MA(q)  model is given by:
y_t=μ+∑_(j=1)^q▒〖θ_j ε_(t-j)+ε_t=μ+θ_1 ε_(t-1)+θ_2 ε_(t-2)+⋯+θ_q ε_(t-q)+ε_t 〗
Here μ is the mean of the series, θj ( j =1,2,...,q) are the model parameters and q is the order of the model. The random shocks are assumed to be a white noise process, i.e. a sequence of independent and identically distributed (i.i.d) random variables with zero mean and a constant variance σ2. Generally, the random shocks are assumed to follow the typical normal distribution. Thus conceptually a moving average model is a linear regression of the current observation of the time series against the random shocks of one or more prior observations. Fitting an MA model to a time series is more complicated than fitting an AR model because in the former one the random error terms are not fore-seeable. 
 
Autoregressive (AR) and moving average (MA) models can be effectively combined together to form a general and useful class of time series models, known as the ARMA models. 
Mathematically an ARMA(p, q) model is represented as: 
y_t=c+ε_t+∑_q^p▒〖φ_i y_(t-i) 〗+∑_(j=1)^q▒〖θ_j ε_(t-j) 〗
Here the model orders p ,q refer to p autoregressive and q moving average terms. 
Usually ARMA models are manipulated using the lag operator notation. The lag or backshift operator is defined as Lyt = yt−1 . Polynomials of lag operator or lag polynomials are used to represent ARMA models as follows:
AR(p) model      : ε_t=φ(L) y_t
MA(q) model     : y_t=θ(L) ε_t
ARMA(p, q) model : φ(L) y_(t=) θ(l) ε_t
φ(L)=1-∑_(i=1)^p▒〖φ_i L_i  and θ(L) 〗=1+∑_(j=1)^q▒〖θ_j L_j 〗
An important property of AR(p) process is invertibility, i.e. an AR(p) process can always be written in terms of an MA(∞) process. Whereas for an MA(q) process to be invertible, all the roots of the equation θ(L) = 0 must lie outside the unit circle. This condition is known as the Invertibility Condition for an MA process.  

3.2.2 Autocorrelation and Partial Autocorrelation Functions (ACF and PACF) 
To determine a proper model for a given time series data, it is necessary to carry out the ACF and PACF analysis. These statistical measures reflect how the observations in a time series are related to each other. For modelling and forecasting purpose it is often useful to plot the ACF and PACF against consecutive time lags. These plots help in determining the order of AR and MA terms. Below we give their mathematical definitions: 
For a time series{x(t),t = 0,1,2,...} the lag k is defined as: 
γ_k=Cov(x_t,x_(t+k) )=E[(x_t-μ)(x_(t+k)-μ)]
The Autocorrelation Coeffient at lag k is defined as: 
p_k=  γ^k/γ^0 
Here μ is the mean of the time series, i.e. μ= E[xt ]. The autocovariance at lag zero i.e. γ0 is the variance of the time series. From the definition it is clear that the autocorrelation coefficient ρk is dimensionless and so is independent of the scale of measurement. Also, clearly −1≤ρk ≤1. Statisticians Box and Jenkins termed γk as the theoretical Autocovariance Function (ACVF) and ρk as the theoretical Autocorrelation Function (ACF).  
Another measure, known as the Partial Autocorrelation Function (PACF) is used to measure the correlation between an observation k period ago and the current observation, after controlling for observations at intermediate lags (i.e. at lags <k ). At lag 1, PACF is same as ACF. 

Normally, the stochastic process governing a time series is unknown and so it is not possible to determine the actual or theoretical ACF and PACF values. Rather these values are to be estimated from the training data, i.e. the known time series at hand. The estimated ACF and PACF values from the training data are respectively termed as sample ACF and PACF. 
The most appropriate sample estimate for the ACVF at lag k is  
c_k=  1/n  ∑_(t=1)^(n-k)▒〖(x_(t )- μ)(x_(t+k)- μ)〗
Then the estimate for the sample ACF at lag k is given 
    
r_k=c_k/c_0 

Here {x(t),t = 0,1,2,.......} is the training series of size n with mean μ.  
 
As explained by Box and Jenkins, the sample ACF plot is useful in determining the type of model to fit to a time series of length N. Since ACF is symmetrical about lag zero, it is only required to plot the sample ACF for positive lags, from lag one onwards to a maximum lag of about N/4. The sample PACF plot helps in identifying the maximum order of an AR process. The methods for calculating ACF and PACF for ARMA models are described. We shall demonstrate the use of these plots for our practical datasets in Chapter 7. 
 
3.2.3 Autoregressive Integrated Moving Average (ARIMA) Models 
The ARMA models, described above can only be used for stationary time series data. However in practice many time series such as those related to socio-economic and business show non-stationary behaviour. Time series, which contain trend and seasonal patterns, are also non-stationary in nature. Thus from application view point ARMA models are inadequate to properly describe non-stationary time series, which are frequently encountered in practice. For this reason the ARIMA model is proposed, which is a generalization of an ARMA model to include the case of non-stationarity as well. 
In ARIMA models a non-stationary time series is made stationary by applying finite differencing of the data points. The mathematical formulation of the ARIMA(p,d,q) model using lag polynomials is given below: 

φ(L) (1-L)^d y_t= θ(L) ε_t  ,i.e

(1-∑_(i=1)^p▒〖φ_i L^i 〗) (1-L)^d y_t= (1+∑_(j=1)^q▒〖φ_j L^j 〗) ε_t

	Here, p, d and q are integers greater than or equal to zero and refer to the order of the autoregressive, integrated, and moving average parts of the model respectively.  
	The integer d controls the level of differencing. Generally d=1 is enough in most cases. 
When d=0, then it reduces to an ARMA(p,q) model.  
	An ARIMA(p,0,0) is nothing but the AR(p) model and ARIMA(0,0,q) is the MA(q) model. 
	 ARIMA(0,1,0), i.e. yt = yt−1 +εt is a special one and known as the Random Walk model. It is widely used for non-stationary data, like economic and stock price series. 
 
A useful generalization of ARIMA models is the Autoregressive Fractionally Integrated Moving Average (ARFIMA) model, which allows non-integer values of the differencing parameter d. ARFIMA has useful application in modelling time series with long memory. In this model the expansion of the term(1− L)d is to be done by using the general binomial theorem. Various contributions have been made by researchers towards the estimation of the general ARFIMA parameters. 
 
3.2.4 Seasonal Autoregressive Integrated Moving Average (SARIMA) Models 
The ARIMA model is for non-seasonal non-stationary data. Box and Jenkins have generalized this model to deal with seasonality. Their proposed model is known as the Seasonal ARIMA (SARIMA) model. In this model seasonal differencing of appropriate order is used to remove non-stationarity from the series. A first order seasonal difference is the difference between an observation and the corresponding observation from the previous year and is calculated as zt = yt − yt−s . For monthly time series s =12 and for quarterly time series s = 4. This model is generally termed as the SARIMA(p,d,q)×(P,D,Q)s model.  
The mathematical formulation of a SARIMA(p,d,q)×(P,D,Q)s model in terms of lag polynomials is given below: 
φ_p (L^s ) φ_p (L) 〖(1-L)〗^d 〖(1-L^s)〗^D y_t=θ_Q (L^s)θ_q (L)ε_t
i.e.φ_p (L^s ) φ_p (L) z_t= θ_Q (L^s)θ_q (L)ε_t
Here zt is the seasonally differenced series.  

3.3 Module-wise Flow Diagram 












 
3.4 Architecture of the project
































3.4 Network Architecture





	

 











 



Chapter 4
Implementation & Results

4.1 System Requirement
	- Hardware Requirements
	Processor : Intel Core i5
	Memory : 8 GB DDR4
	Disk : 500GB
	Graphics : Intel Integrated Graphics
	- Software Requirements
	Anaconda Distribution for python
	Operating system platform independent
	Jupyter Notebook
	Spyder
	Python 3.8 

4.2 Dependent Libraries Description (Requirements.txt)
Python's standard library covers a wide range of modules. Everything from modules that are as much a part of the Python language as the types and statements defined by the language specification, to obscure modules that are probably useful only to a small number of programs. This section describes a number of fundamental standard library modules. Any larger Python program is likely to use most of these modules, either directly or indirectly.

4.2.1 Pandas
Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with structured (tabular, multidimensional, potentially heterogeneous) and time series data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis / manipulation tool available in any language. It is already well on its way toward this goal. Pandas is well suited for many different kinds of data:
	Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet
	Ordered and unordered (not necessarily fixed-frequency) time series data.
	Arbitrary matrix data (homogeneously typed or heterogeneous) with row and column labels
	Any other form of observational / statistical data sets. The data actually need not be labelled at all to be placed into a pandas data structure
4.2.2 NumPy
NumPy is a Python package. It stands for 'Numerical Python'. It is a library consisting of multidimensional array objects and a collection of routines for processing of array.
Numeric, the ancestor of NumPy, was developed by Jim Hugunin. Another package Numarray was also developed, having some additional functionalities. In 2005, Travis Oliphant created NumPy package by incorporating the features of Numarray into Numeric package. There are many contributors to this open source project.
Operations using NumPy using NumPy, a developer can perform the following operations :
	Mathematical and logical operations on arrays.
	Fourier transforms and routines for shape manipulation.
	Operations related to linear algebra. NumPy has in-built functions for linear algebra and random number generation.
4.2.3 Seaborn
Seaborn is a library for making statistical graphics in Python. It is built on top of matplotlib and closely integrated with pandas data structures.
Here is some of the functionality that seaborn offers:
	A dataset-oriented API for examining relationships between multiple variables 
	Specialized support for using categorical variables to show observations or aggregate statistics.
	Options for visualizing univariate or bivariate distributions and for comparing them between subsets of data
	Automatic estimation and plotting of linear regression models for different kinds dependent variables
	Convenient views onto the overall structure of complex datasets
	High-level abstractions for structuring multi-plot grids that let you easily build complex visualizations
	Concise control over matplotlib figure styling with several built-in themes 
	Tools for choosing color palettes that faithfully reveal patterns in your data
Seaborn aims to make visualization a central part of exploring and understanding data. Its dataset-oriented plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots.
4.2.4 Matplotlib 
Matplotlib is one of the most popular Python packages used for data visualization. It is a cross-platform library for making 2D plots from data in arrays. Matplotlib is written in Python and makes use of NumPy, the numerical mathematics extension of Python. It provides an object-oriented API that helps in embedding plots in applications using Python GUI toolkits such as PyQt, WxPythonotTkinter. It can be used in Python and IPython shells, Jupyter notebook and web application servers also.
Matplotlib has a procedural interface named the Pylab, which is designed to resemble MATLAB, a proprietary programming language developed by MathWorks. Matplotlib along with NumPy can be considered as the open source equivalent of MATLAB.
Matplotlib was originally written by John D. Hunter in 2003. The current stable version is 2.2.0 released in January 2018.
4.2.5 Stats Models
As its name implies, statsmodels is a Python library built specifically for statistics. Statsmodels is built on top of NumPy, SciPy, and matplotlib, but it contains more advanced functions for statistical testing and modelling that you won't find in numerical libraries like NumPy or SciPy. Python StatsModels allows users to explore data, perform statistical tests and estimate statistical models. It is supposed to complement to SciPy’s stats module. It is part of the Python scientific stack that deals with data science, statistics and data analysis.
It also uses Pandas for data handling and Patsy for R-like formula interface. It takes its graphics functions from matplotlib. It is known to provide statistical background for other python packages.
Originally, Jonathan Taylor wrote the models module of scipy.stats. It was part of scipy for some time but was removed later.
It was tested, corrected and improved during the Google Summer of Code 2009 and launched as a new package we know as StatsModels.
New models, plotting tools and statistical models are being introduced continuously developed and introduced by the StatsModels development team.
4.2.6 SCIKIT-LEARN:
Scikit-learn (formerly scikits.learn and also known as sklearn) is a free software learning library for the Python Programming Language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
Scikit-learn is an open source Python library that implements a range of machine learning, pre-processing, cross-validation and visualization algorithms using a unified interface.
Important features of scikit-learn:
	Simple and efficient tools for data mining and data analysis. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means, etc.
	Accessible to everybody and reusable in various contexts.
	Built on the top of NumPy, SciPy, and matplotlib.
	Open source, commercially usable – BSD license.



4.3 Project Folder Structure

The structure of the project is as follows in spyder :



4.4 Module-wise Description with Source Code
4.4.1 Data Input Module
Groundwater is an important source of drinking and agriculture in our country. But this is hard to be explored and analysed being flowing in sub-surface. The level of water below ground is temporal and dynamic in nature. It is mainly controlled by rainfall pattern in relation to the aquifer material. Central Ground Water Board (CGWB) and State Ground water departments have implanted monitoring wells and measure ground water level regularly.
The GW Level module provides the available water level data for 29 states & 7 UTs (excluding some hilly regions of north and north east India) for the period 1994 to 2015 as received from CGWB. It contains a layer having locations of more than 30,000 observation wells with 16 attributes (11 static and 5 temporal). Further, water level data is grouped according to four seasons viz.
	Post-monsoon Rabi (January to March)
	Pre monsoon (April to June)
	Monsoon (July to September)
	Post-monsoon Kharif (October to December)
The user can view season wise water level of individual or multiple wells in graphical form as per predefined area as well as administrative/basin boundary. One can also assess the water level changes and compare its dynamic fluctuation in spatial and temporal domain through this sub info system.

Source Code 
To load raw data
	# Download csv file from resources and put it in working directory
	raw_data = pd.read_csv('India_GWL.csv')
 to exploring raw data
	raw_data.tail()
 

 4.4.2 Data Analysis Module
Data analysis is performed on the input data to get meaningful insights about the dataset, analyse various types of data in the dataset, the relationships between various data features, deal with missing values etc. Predictive analysis is performed to use data, statistical algorithms and machine learning techniques to identify the likelihood of future outcomes based on historical data. The goal is to go beyond knowing what has happened to providing a best assessment of what will happen in the future.

Source Code :
Null Treatment : Before we dive into code, it’s important to understand the sources of missing data. Here’s some typical reasons why data is missing:
	User forgot to fill in a field.
	Data was lost while transferring manually from a legacy database.
	There was a programming error.
	Users chose not to fill out a field tied to their beliefs about how the results would be used or interpreted.
As you can see, some of these sources are just simple random mistakes. Other times, there can be a deeper reason why data is missing.
	#heatmap to check for null values in the dataframe
	sns.heatmap(dataframe.isnull(),yticklabels=False,cbar=False,cmap='viridis')


Replacing null values with median for POMRB, PREMON, MONSOON & POMKH
	dataframe.POMRB.fillna(dataframe.POMRB.quantile(0.5),inplace = True)
	dataframe.PREMON.fillna(dataframe.PREMON.quantile(0.5),inplace = True)
	dataframe.MONSOON.fillna(dataframe.MONSOON.quantile(0.5),inplace = True)
	dataframe.POMKH.fillna(dataframe.POMKH.quantile(0.5),inplace = True)
Testing if the null values have been replaced
	dataframe.isnull().sum()
Outlier Treatment :  An outlier is any data point which differs greatly from the rest of the observations in a dataset. There are a plethora of reasons why outliers exist. Perhaps an analyst made an error in the data entry, or the machine threw up an error in measurement, or the outlier could even be intentional! Some people do not want to disclose their information and hence input false information in forms.
Outliers are of two types: Univariate and Multivariate. A univariate outlier is a data point that consists of extreme values in one variable only, whereas a multivariate outlier is a combined unusual score on at least two variables. Suppose you have three different variables – X, Y, Z. If you plot a graph of these in a 3-D space, they should form a sort of cloud. All the data points that lie outside this cloud will be the multivariate outliers.
Plot a box plot to see the outliers :
	 dataframe.POMRB.plot(kind='box')
 

	#replacing outliers with lower quantile & upper quantile
	dataframe.POMRB.clip(dataframe.POMRB.quantile(0),dataframe.POMRB.quantile(0.99),inplace=True)
	dataframe.POMRB.plot(kind = 'box')
 
4.4.3 Data Visualization Module
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.
In the world of Big Data, data visualization tools and technologies are essential to analyse massive amounts of information and make data-driven decisions.
Our eyes are drawn to colors and patterns. We can quickly identify red from blue, square from circle. Our culture is visual, including everything from art and advertisements to TV and movies.
Data visualization is another form of visual art that grabs our interest and keeps our eyes on the message. When we see a chart, we quickly see trends and outliers. If we can see something, we internalize it quickly. It’s storytelling with a purpose. If you’ve ever stared at a massive spreadsheet of data and couldn’t see a trend, you know how much more effective a visualization can be.
Visualising the data on District level
	a = df['LEVEL'][df.DISTRICT == "Ludhiana"]
	plt.plot(a.reset_index()['LEVEL'])
	sns.regplot(x = a.reset_index().index.values , y = a , order = 3 , marker = 'o' , color = 'crimson')
 	

4.4.4 Exploratory data analysis
In data mining, Exploratory Data Analysis (EDA) is an approach to analysing datasets to summarize their main characteristics, often with visual methods. EDA is used for seeing what the data can tell us before the modelling task. It is not easy to look at a column of numbers or a whole spreadsheet and determine important characteristics of the data. It may be tedious, boring, and/or overwhelming to derive insights by looking at plain numbers. Exploratory data analysis techniques have been devised as an aid in this situation.
Exploratory data analysis is generally cross-classified in two ways. First, each method is either non-graphical or graphical. And second, each method is either univariate or multivariate (usually just bivariate).
Source Code
ACF: ACF is an (complete) auto-correlation function which gives us values of auto-correlation of any series with its lagged values. We plot these values along with the confidence band and tada! We have an ACF plot. In simple terms, it describes how well the present value of the series is related with its past values. A time series can have components like trend, seasonality, cyclic and residual. ACF considers all these components while finding correlations hence it’s a ‘complete auto-correlation plot’.
	plot_acf(df['LEVEL'][df.DISTRICT == 'Ludhiana']).show()
 
PACF: PACF is a partial auto-correlation function. Basically instead of finding correlations of present with lags like ACF, it finds correlation of the residuals (which remains after removing the effects which are already explained by the earlier lag(s)) with the next lag value hence ‘partial’ and not ‘complete’ as we remove already found variations before we find the next correlation. So if there is any hidden information in the residual which can be modelled by the next lag, we might get a good correlation and we will keep that next lag as a feature while modelling. Remember while modelling we don’t want to keep too many features which are correlated as that can create multicollinearity issues. Hence we need to retain only the relevant features.
	plot_pacf(df['LEVEL'][df.DISTRICT == 'Ludhiana']).show()



Seasonal Decompose : The statsmodels library provides an implementation of the naive, or classical, decomposition method in a function called seasonal_decompose(). It requires that you specify whether the model is additive or multiplicative.
Both will produce a result and you must be careful to be critical when interpreting the result. A review of a plot of the time series and some summary statistics can often be a good start to get an idea of whether your time series problem looks additive or multiplicative.
The seasonal_decompose() function returns a result object. The result object contains arrays to access four pieces of data from the decomposition.
For example, the snippet below shows how to decompose a series into trend, seasonal, and residual components assuming an additive model.
The result object provides access to the trend and seasonal series as arrays. It also provides access to the residuals, which are the time series after the trend, and seasonal components are removed. Finally, the original or observed data is also stored.
	seasonal_decompose(df['LEVEL'][df.DISTRICT == 'Ludhiana'], model = 'additive').plot().show()
 


4.4.5 Graphical user interface (GUI) Development 
Python provides various options for developing graphical user interfaces (GUIs). Most important are listed below.
	Tkinter − Tkinter is the Python interface to the Tk GUI toolkit shipped with Python. We would look this option in this chapter.
	wxPython − This is an open-source Python interface for wxWindows 
	JPython − JPython is a Python port for Java which gives Python scripts seamless access to Java class libraries on the local machine
Tkinter : Tkinter is the standard GUI library for Python. Python when combined with Tkinter provides a fast and easy way to create GUI applications. Tkinter provides a powerful object-oriented interface to the Tk GUI toolkit.
Creating a GUI application using Tkinter is an easy task. All you need to do is perform the following steps −
	Import the Tkinter module.
	Create the GUI application main window.
	Add one or more of the above-mentioned widgets to the GUI application.
	Enter the main event loop to take action against each event triggered by the user.
4.5 Results
The SARIMA model is being trained on data from 1990 to 2015. After running the code a pop up will appear in which we can choose to select the State and Districts from the dropdown menu. 
 
After running we get the output as a graph that provide us with the prediction of next 5 years, that is up to 2020.
 
4.5.1 Computational Results
RMSE : Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.
 

 

References

	[1]  Burges, C.J.C., “A tutorial on support vector machines for pattern recognition”, Data Mining and Knowledge Discovery, 2 (1998), pages: 121-167. 
	[2]  C. Chatfield, “Model uncertainty and forecast accuracy”, J. Forecasting 15 (1996), pages: 495–508. 
	[3]  C. Hamzacebi, “Improving artificial neural networks’ performance in seasonal time series forecasting”, Information Sciences 178 (2008), pages: 4550-4559. 
	[4]  F. Girosi, M. Jones, and T. Poggio, “Priors, stabilizers and basic functions: From regularization to radial, tensor and additive splines.” AI Memo No: 1430, MIT AI Lab, 1993. 
	[5]  G. Zhang, B.E. Patuwo, M.Y. Hu, “Forecasting with artificial neural networks: The state of the art”, International Journal of Forecasting 14 (1998), pages: 35-62. 
	[6]  G.E.P. Box, G. Jenkins, “Time Series Analysis, Forecasting and Control”, Holden-Day, San Francisco, CA, 1970. 
	[7]  G.P. Zhang, “A neural network ensemble method with jittered training data for time series forecasting”, Information Sciences 177 (2007), pages: 5329–5346. 
	[8]  G.P. Zhang, “Time series forecasting using a hybrid ARIMA and neural network model”, Neurocomputing 50 (2003), pages: 159–175. 
	[9]  H. Park, “Forecasting Three-Month Treasury Bills Using ARIMA and GARCH Models”, Econ 930, Department of Economics, Kansas State University, 1999. 
	[10]  H. Tong, “Threshold Models in Non-Linear Time Series Analysis”, Springer-Verlag, New York, 1983. 


