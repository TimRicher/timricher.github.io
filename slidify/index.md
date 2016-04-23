---
title       : Factor Impacting Automobile MPG
subtitle    : John Hopkins University Data Products Class
author      : Tim Richer
job         : 
framework   : io2012        # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : []            # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}
knit        : slidify::knit2slides
---



<h2>My Shiny App lets you choose different factors to include in a linear model.</h2>
<h3>The linear model shows the impact of the factor on miles per gallon.</h3>
<h3>The plot renders the residuals versus the fitted values. </h3>
<p>
<b>The mtcars dataset has a number of factors that impact mpg:</b> <br>
<b>cyl</b>   - Number of Cylinders in the Engine  <br>
<b>disp</b>  - Displacement of the Engine (cu.in.) <br>
<b>hp</b>    - Gross Horsepower <br>
<b>drat</b>  - Rear Axil Ratio <br>
<b>wt</b>    - Weight (lb/1000) <br>
<b>qsec</b>  - Time in the Quarter Mile <br>
<b>v/s</b>   - V/S <br>
<b>am</b>    - Transmission (0=Automatic, 1=Manual) <br>
<b>gear</b>  - Number of Gears <br>
<b>carb</b>  - Number of Carburators <br>
</p>

--- .class #id 

<p>
For citizen data scientists, <b>that are just becoming familiar with the power of R </b>,<br>
My Shiny App lets them interactively explore factors, in the mtcars dataset, that impact mpg.<br>
The app allows many factors to be simultaneously considered, not just one at a time. <br>
Like the graph below: <br>


![plot of chunk unnamed-chunk-1](assets/fig/unnamed-chunk-1-1.png)

--- .class #id 

<p>
The plots rendering the residuals versus the fitted values, like the graphs below, <br>
allow for interesting interaction when different factors are included. <br>
This was the genesis of My Shiny App. <br>
![plot of chunk unnamed-chunk-2](assets/fig/unnamed-chunk-2-1.png)

--- .class #id 

My Shiny App lets you choose different factors to include in a linear model.
The linear model shows the impact of the factor(s) on miles per gallon.

## Shiny Application and Source

Shiny App Link:
https://timricherjhu.shinyapps.io/MyProject/

Link to GitHib Shiny Source code:
https://github.com/TimRicher/Data_Products




