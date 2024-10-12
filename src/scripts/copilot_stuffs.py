# Copilot code to generate IRFs
# import numpy as np
# from scipy.linalg import inv

# def calculate_irfs(VAR_bet, VAR_det, VAR_res, VAR_m, VAR_T, horizon=10):
#     """
#     Calculate Impulse Response Functions (IRFs) using the proxy SVAR model output.

#     Parameters:
#     VAR_bet (np.ndarray): The VAR coefficient matrix.
#     VAR_det (np.ndarray): The deterministic terms matrix.
#     VAR_res (np.ndarray): The residuals matrix.
#     VAR_m (np.ndarray): The external instruments matrix.
#     VAR_T (int): The sample size.
#     horizon (int): The number of periods to calculate the IRFs for.

#     Returns:
#     np.ndarray: The calculated IRFs.
#     """
#     # Narrative Identification
#     AL = VAR_bet[:-VAR_det.shape[1], :].T
#     VAR_Gamma = (VAR_res.T @ VAR_m) / VAR_T

#     # Calculate the structural shocks
#     B = inv(AL) @ VAR_Gamma

#     # Initialize IRFs array
#     num_vars = VAR_bet.shape[0]
#     irfs = np.zeros((num_vars, num_vars, horizon))

#     # Initial impact
#     irfs[:, :, 0] = B

#     # Calculate IRFs for each period
#     for t in range(1, horizon):
#         irfs[:, :, t] = AL @ irfs[:, :, t-1]

#     return irfs

# # Example usage:
# # Assuming VAR_bet, VAR_det, VAR_res, VAR_m, and VAR_T are already defined as NumPy arrays or appropriate variables
# # horizon = 10  # Number of periods to calculate the IRFs for
# # irfs = calculate_irfs(VAR_bet, VAR_det, VAR_res, VAR_m, VAR_T, horizon)
# # print(irfs)