# Example 1: Examining the Model Summary
import statsmodels.api as sm

X = [0.2, 0.32, 0.38, 0.41, 0.43]
y = [0.1, 0.15, 0.4,  0.6,  0.44]

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# Example 2: Simple Least Squares Regression