# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:49:30 2024

@author: Yang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from plotnine import ggplot, aes, geom_line, labs,theme,facet_grid


def main():
    path = r"Submit_2024-03-02_20-11-39.log"

    import re

    epoch_num = []
    iteration_num = []
    classification_loss = []
    regression_loss = []
    running_loss = []

         

    with open(path, 'r') as file:
        for line in file:
            match = re.match(r'Epoch: (\d+) \| Iteration: (\d+) \| Classification loss: (\d+\.\d+) \| Regression loss: (\d+\.\d+) \| Running loss: (\d+\.\d+)', line)
            if match:
                epoch_num.append(int(match.group(1)))
                iteration_num.append (int(match.group(2)))
                classification_loss.append(float(match.group(3)))
                regression_loss.append(float(match.group(4)))
                running_loss.append( float(match.group(5)))

    summary = pd.DataFrame({
        'Epoch':epoch_num,
        "Iteration":iteration_num,
        "Classification loss":classification_loss,
        "Regression loss":regression_loss,
        "Running loss":running_loss})    

    summary['Iteration'] = (summary['Iteration']) + summary['Epoch']*500

    plt.figure()
    plt.plot(summary['Iteration'], summary['Classification loss'],color='gray')
    plt.xlabel('Iteration', fontname='Times New Roman')
    plt.ylabel('Classification Loss', fontname='Times New Roman')
    plt.title('Classification Loss vs. Iteration', fontname='Times New Roman')
    plt.grid()
    # Change font for tick labels
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')

    # Display the plot
    plt.show()
    plt.savefig('Classification Loss vs. Iteration.png')


    plt.figure()
    plt.plot(summary['Iteration'], summary['Regression loss'],color='gray')
    plt.xlabel('Iteration', fontname='Times New Roman')
    plt.ylabel('Regression Loss', fontname='Times New Roman')
    plt.title('Regression Loss vs. Iteration', fontname='Times New Roman')
    plt.grid()
    # Change font for tick labels
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')

    # Display the plot
    plt.show()
    plt.savefig('Regression Loss vs. Iteration.png')



    plt.figure()
    plt.plot(summary['Iteration'], summary['Running loss'],color='grey')
    plt.xlabel('Iteration', fontname='Times New Roman')
    plt.ylabel('Running Loss', fontname='Times New Roman')
    plt.title('Running Loss vs. Iteration', fontname='Times New Roman')
    plt.grid()
    # Change font for tick labels
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')

    # Display the plot
    plt.show()
    plt.savefig('Running Loss vs. Iteration.png')


if __name__ == '__main__':
    main()
#%%


# # Create the plot using plotnine
# plot = ggplot(summary, aes(x='Iteration', y='Classification loss')) + \
#     geom_line() + \
#     labs(x='Iteration', y='Classification Loss', title='Classification Loss')+\
# plot.show()
    

# plot = ggplot(summary, aes(x='Iteration', y='Classification loss')) + \
#     geom_line() + \
#     labs(x='Iteration', y='Classification Loss', title='Classification Loss')+\
# plot.show()


# # Display the plot
# print(plot)