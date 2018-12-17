#!/usr/bin/python

"""
Import packages/libraries
"""
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from my_config import Config as cfg


def main():
    main_title = 'Advertisement Sales Trends and Predictions'

    # read advertisement data sample into DataFrame
    ad = pd.read_csv(
        '{pp}/{pd}/{fn}'.format(
            pp=cfg.PATH_PARENT
            , pd=cfg.PATH_DATA
            , fn=cfg.DATA_ADV
        )
        , index_col=0
    )

    # show the first 5 rows of the imported data
    #print(ad.head())
    # print the shape of the DataFrame
    #print(ad.shape)

    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)

    # adding the scatterplots to the grid
    ad.plot(kind='scatter', x='TV', y='sales', ax=axs[0], figsize=(16, 8))
    ad.plot(kind='scatter', x='Radio', y='sales', ax=axs[1])
    ad.plot(kind='scatter', x='Newspaper', y='sales', ax=axs[2])

    #plt.show()

    """
    Simple linear regression model
    >> $$y = \beta_0 + \beta_1x$$
    
    * $$y$$ is the predicted numerical value (response) -> *sales*
    * x is the feature
    * $$\beta_0$$ is called the *intercept*, which is the value of $$y$$ when $$x = 0$$
    * $$\beta_1$$ is the coefficient of the feature $$x$$ also called the *slope*, which represent the change in $$y$$ divided by the change in $$x$$
        $$\beta_1 = \frac{\delta y}{\delta x}$$
    
    create a fitted model that represent the least squares line
    
    TV's Output:
    TV sales's coefficient:
    Intercept    7.032594
    TV           0.047537
    dtype: float64
    """
    tv_lm = smf.ols(formula='sales~TV', data=ad).fit()
    radio_lm = smf.ols(formula='sales~Radio', data=ad).fit()
    newspaper_lm = smf.ols(formula='sales~Newspaper', data=ad).fit()
    # show the trained model coefficients
    #print("TV sales's coefficient:\n{pa}".format(pa=tv_lm.params))
    #print("Radio sales's coefficient:\n{pa}".format(pa=radio_lm.params))
    #print("Newspaper sales's coefficient:\n{pa}".format(pa=newspaper_lm.params))

    """
    A unit increase in the feature (TV ad) spending is associated with 0.047537 unit increase in Sales (response).
    In other words, an additional $100 spent on TV ads is associated with an increase in sales of 4.7537
    
    TV's Output:
    7.032593549127694 | 0.047536640433019764
    """
    b0 = tv_lm.params[0]
    b1 = tv_lm.params[1]
    #print("{b0} | {b1}".format(b0=b0, b1=b1))

    """
    Make a manual prediction of how much sales will increase from $50,000 of TV advertising
    
    Output:
    2383.864615200116
    """
    sales = b0 + (b1 * 50000)
    #print(sales)

    """
    creating a new Pandas DataFrame to match Statsmodels interface expectations
    
    TV's Output:
          TV
    0  50000
    """
    new_tv_ad_spending = pd.DataFrame({'TV': [50000]})
    #print(new_tv_ad_spending.head())

    """
    use the model to make predictions on a new value
    
    TV's Output:
    0    2383.864615
    """
    preds = tv_lm.predict(new_tv_ad_spending)
    #print(preds)

    """
    least square line. In order to draw the line, need two points: minimum and maximum values from feature
    
    TV's Output:
          TV
    0    0.7
    1  296.4    
    """
    tv_min_max = pd.DataFrame({'TV': [ad['TV'].min(), ad['TV'].max()]})
    radio_min_max = pd.DataFrame({'Radio': [ad['Radio'].min(), ad['Radio'].max()]})
    newspaper_min_max = pd.DataFrame({'Newspaper': [ad['Newspaper'].min(), ad['Newspaper'].max()]})
    #print(tv_min_max)
    #print(radio_min_max)
    #print(newspaper_min_max)

    """
    predictions for x min and max values
    
    TV's Output:
    0     7.065869
    1    21.122454
    dtype: float64    
    """
    tv_predictions = tv_lm.predict(tv_min_max)
    radio_predictions = radio_lm.predict(radio_min_max)
    newspaper_predictions = newspaper_lm.predict(newspaper_min_max)
    #print(tv_predictions)
    #print(radio_predictions)
    #print(newspaper_predictions)

    # plotting the least squares line
    #plt.plot(tv_min_max, tv_predictions, color='red', linewidth=2)
    axs[0].plot(tv_min_max, tv_predictions, color='red', linestyle='-', linewidth=2)
    axs[0].grid(True)

    axs[1].plot(radio_min_max, radio_predictions, color='red', linestyle='-', linewidth=2)
    axs[1].grid(True)

    axs[2].plot(newspaper_min_max, newspaper_predictions, color='red', linestyle='-', linewidth=2)
    axs[2].grid(True)

    fig.tight_layout()
    fig.suptitle(main_title, fontsize=18, fontweight='bold')
    #fig.legend(loc='upper right')

    fig_mgr = plt.get_current_fig_manager()
    fig_mgr.window.showMaximized()

    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == '__main__':
    main()
