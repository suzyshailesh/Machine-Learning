import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize


def random_dataset(n):
    #generate n random x values and n random y values
    x = np.random.rand(n)
    y = np.random.rand(n)

    #make x and y numpy arrays for ease of use
    x = np.array(list(x))
    y = np.array(list(y))

    return [x, y]

def my_linear_model(x, y):
    #from-scratch linear regression model

    #calculate slope of linear regression model, beta 1
    cov = np.cov(x, y)[0][1]
    var = np.var(x)
    beta_1 = cov/var

    #calculate intercept of linear regression model, beta 0
    beta_0 = np.mean(y) - beta_1 * np.mean(x)

    #calculate f(x) values for linear regression model
    #based on the linear regression formula f(x) = beta_0 + beta_1*x
    f_x = [beta_0+beta_1*i for i in x]

    #print model parameters
    print("slope: " + str(beta_1))
    print("intercept: " + str(beta_0))

    #calculate performance metrics for regression
    performance_metrics(x, y, f_x, beta_1)

    return beta_1, beta_0

def performance_metrics(x, y, f_x, beta_1):
    #calculate performance metrics for from-scratch linear regression

    #various calculations
    epsilons = y-f_x
    sigma_sq = np.var(epsilons)

    mean_x = np.mean(x)
    x_minus_mean = (x - mean_x)**2
    summation_x = np.sum(x_minus_mean)

    mean_y = np.mean(y)
    y_minus_mean = (y - mean_y)**2
    summation_y = np.sum(y_minus_mean)

    std_beta_1_sq = sigma_sq/summation_x

    std_beta_0_sq = sigma_sq*((1/len(x))+((mean_x**2)/summation_x))

    #compute t stat
    t_stat = beta_1/np.sqrt(std_beta_1_sq)

    summation_f_x = np.sum(epsilons**2)

    #compute rse
    rse = np.sqrt((1/(len(f_x)-2))*summation_f_x)

    #compute r squared
    r_squared = 1 - (summation_f_x/summation_y)

    #print perfomance metrics
    print("t-stat: " + str(t_stat))
    print("rse: " + str(rse))
    print("r-squared: " + str(r_squared))

    return 1

def scikit_linear_model(x, y):
    #sklearn linear regression model

    #split dataset into test and train values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #fit linear regression
    reg = LinearRegression().fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

    #use test dataset to predict values
    y_pred = reg.predict(x_test.reshape(-1, 1))

    #find and print model parameters
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    print("slope: " + str(slope))
    print("intercept: " + str(intercept))

    #calculate residual standard error
    residuals = y_test - y_pred
    df = len(x)-2 
    rss = np.sum(residuals**2)  # Residual sum of squares
    rse = np.sqrt(rss/df)  # Residual standard error

    #calculate r-squared 
    r_squared = r2_score(y_test, y_pred)

    #calculate t-stat
    t_stat = reg.coef_[0]/(rse/np.sqrt(np.sum((x-np.mean(x))**2)))

    #print performance metrics
    print("t-stat: " + str(t_stat[0]))
    print("rse: " + str(rse))
    print("r-squared: " + str(r_squared))

    return slope, intercept

def g_optimize(x, y):
    #define loss function
    def g(params, x, y):
        beta_0, beta_1 = params
        return np.sum((y - (beta_0 - beta_1 * x)) ** 2)

    #initial guess for the parameters
    initial_params = np.array([0, 0])
    result = minimize(g, initial_params, args=(x, y))

    #print the model parameters
    params = result.x
    intercept = params[0]
    slope = params[1]
    print("slope: " + str(slope))
    print("intercept: " + str(intercept))

    #return the optimized parameters
    return slope, intercept

def l1_optimize(x, y):
    #define loss function
    def l1(params, x, y):
        beta_0, beta_1 = params
        return np.sum(np.abs(y - (beta_0 - beta_1 * x)))

    #initial guess for the parameters
    initial_params = np.array([0, 0])
    result = minimize(l1, initial_params, args=(x, y))

    #print the model parameters
    params = result.x
    intercept = params[0]
    slope = params[1]
    print("slope: " + str(slope))
    print("intercept: " + str(intercept))

    #return the optimized parameters
    return slope, intercept

def main():
    #create random dataset of size 100
    x, y = random_dataset(100)

    #fit from-scratch linear regression model and display model parameters and performance metrics
    print("My Linear Regression model:")
    my_regression_slope, my_regression_intercept = my_linear_model(x, y)

    #fit sklearn linear regression model and display model parameters and performance metrics
    print("Sklearn Linear Regression Model:")
    sklearn_regression_slope, sklearn_regression_intercept = scikit_linear_model(x, y)
    
    #optimization
    print("Optimization:")
    print("g(beta_0, beta_1):")
    g_optimization_slope, g_optimization_intercept = g_optimize(x, y)
    print("l1(beta_0, beta_1):")
    l1_optimization_slope, l1_optimization_intercept = l1_optimize(x, y)

    #show plot with all models
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.scatter(x, y)
    plt.plot(x, my_regression_intercept + my_regression_slope * x, label = "My Linear Regression")
    plt.plot(x, sklearn_regression_intercept + sklearn_regression_slope * x, label = "Sklearn Linear Regression Model")
    plt.plot(x, g_optimization_intercept + g_optimization_slope * x, label = "G(beta_0, beta_1) Optimization")
    plt.plot(x, l1_optimization_intercept + l1_optimization_slope * x, label = "L1 Loss Function")
    plt.legend()
    plt.show()

    #print conclusions
    print("Conclusions:")
    print("All of the different models give very similar results. None are great at estimating the " + \
          "values, but it is difficult to do this with a random dataset. All models estimate a line " + \
          "that has a slope of around 0 and an intercept of around 0.5. This creates a horizontal line " + \
          "across the middle of the scatterplot. These results are as expected for a random dataset.")

    return 1



if __name__ == "__main__":
    main()

