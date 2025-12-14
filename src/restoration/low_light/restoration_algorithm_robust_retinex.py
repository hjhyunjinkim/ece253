import numpy as np
import cv2
import os
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg, LinearOperator
from scipy.fft import fft2, ifft2

# Implemented from paper Structure-Revealing Low-Light Image Enhancement Via Robust Retinex Model
# https://ieeexplore.ieee.org/document/8304597
class RobustRetinex:
    def __init__(self, beta=0.01, omega=0.01, delta=10., gamma_correction=2.2):
        """
        Initializes the Robust Retinex model parameters.
        Parameters based on Section IV-B-2 (Noise Suppression).
        """
        self.beta = beta
        self.omega = omega
        self.delta = delta
        self.gamma = gamma_correction
        self.iterations = 10
        self.epsilon = 1e-3 # Convergence threshold

    def _get_gradients(self, img):
        """
        Calculates first-order gradients (Forward difference).
        """
        h, w = img.shape
        # Gradient x (horizontal)
        dx = np.roll(img, -1, axis=1) - img
        dx[:, -1] = 0
        # Gradient y (vertical)
        dy = np.roll(img, -1, axis=0) - img
        dy[-1, :] = 0
        return dx, dy

    def _divergence(self, dx, dy):
        """
        Calculates divergence (Transpose of Gradient operator).
        D^T * x
        """
        # Inverse of dx (backward difference)
        dx_back = np.roll(dx, 1, axis=1) - dx
        dx_back[:, 0] = -dx[:, 0]
        
        # Inverse of dy
        dy_back = np.roll(dy, 1, axis=0) - dy
        dy_back[0, :] = -dy[0, :]
        
        return dx_back + dy_back

    def _generate_laplacian_matrix(self, h, w):
        """
        Constructs the sparse Laplacian matrix (D^T * D) for the image size.
        Used in Eq 14 and 16 solutions.
        """
        size = h * w
        # 5-point Laplacian structure
        D1 = -4 * np.ones(size)
        D2 = np.ones(size)
        D3 = np.ones(size)
        D4 = np.ones(size)
        D5 = np.ones(size)
        
        D2[0::w] = 0 
        D3[w-1::w] = 0

        diagonals = [D1, D2, D3, D4, D5]
        offsets = [0, -1, 1, -w, w]
        
        lap = spdiags(diagonals, offsets, size, size)
        
        return -lap
    
    def _compute_laplacian_eigenvalues(self, h, w):
        """
        Computes the eigenvalues of the Laplacian matrix in the Fourier domain.
        This represents the denominator of the FFT preconditioner.
        """
        # Laplacian kernel (D^T * D)
        kernel = np.zeros((h, w))
        
        # Center
        kernel[0, 0] = 4

        # Neighbors (using periodic boundary conditions for FFT approximation)
        kernel[0, 1] = -1
        kernel[0, -1] = -1
        kernel[1, 0] = -1
        kernel[-1, 0] = -1
        
        # Optical Transfer Function (OTF) / Eigenvalues
        return np.real(fft2(kernel))

    def _cal_guidance_G(self, I):
        """
        Calculates the adjusted gradient matrix G.
        Implements Equations (7), (8), (9).
        """
        dx, dy = self._get_gradients(I)
        mag = np.abs(dx) + np.abs(dy)
        
        # Eq 9: Thresholding small gradients (noise)
        noise_threshold = 0.01 # epsilon in Eq 9
        mask = mag > noise_threshold
        
        dx_hat = dx * mask
        dy_hat = dy * mask
        mag_hat = mag * mask

        # Eq 8: Amplification factor K
        lam = 10.0
        sigma = 10.0
        K = 1 + lam * np.exp(-mag_hat / sigma)

        # Eq 7: G = K * Gradient
        Gx = K * dx_hat
        Gy = K * dy_hat
        
        return Gx, Gy

    def _solve_subproblem(self, diag_A, laplacian, rhs, weight):
        """
        Solves (diag_A + weight * Laplacian) * x = rhs
        Uses Preconditioned Conjugate Gradient (PCG) with FFT approximation.
        """
        h, w = rhs.shape
        size = h * w
        rhs_flat = rhs.flatten()
        diag_flat = diag_A.flatten()

        # Laplacian Eigenvalues
        lap_eig = self._compute_laplacian_eigenvalues(h, w)
        mean_diag = np.mean(diag_flat)
        
        # Preconditioner Operator M^(-1)
        # M approximates A as: (mean_diag * I + weight * Laplacian)
        denominator = mean_diag + weight * lap_eig
        
        def apply_preconditioner(r_vector):
            r_img = r_vector.reshape(h, w)
            
            # Solve M * z = r in Fourier Domain
            # z = IFFT( FFT(r) / (gamma + weight * eigenvalues) )
            r_fft = fft2(r_img)
            z_fft = r_fft / (denominator + 1e-10) # Avoid division by zero
            z_img = np.real(ifft2(z_fft))
            
            return z_img.flatten()
        M = LinearOperator((size, size), matvec=apply_preconditioner)

        # LinearOperator for the actual system A
        def apply_A(x_vector):      # A = diag_A + weight * laplacian
            x = x_vector
            term1 = diag_flat * x
            term2 = weight * (laplacian @ x)
            return term1 + term2
        A_op = LinearOperator((size, size), matvec=apply_A)

        # PCG run
        x, info = cg(A_op, rhs_flat, M=M, rtol=1e-4, maxiter=50) # Reduced maxiter, PCG converges faster

        if info > 0:
            print(f"  [PCG] Convergence warning: code {info}")
            
        return x.reshape(h, w)

    def _shrinkage(self, x, epsilon):
        """
        Soft thresholding operator for T update.
        Eq 20.
        """
        return np.sign(x) * np.maximum(np.abs(x) - epsilon, 0)

    def enhance(self, img_path):
        # Read and convert to HSV
        img_bgr = cv2.imread(img_path)
        img_bgr = img_bgr.astype(np.float32) / 255.0
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Extract V channel (I) normalized [0,1]
        I = img_hsv[:, :, 2]
        h, w = I.shape
        
        # Pre-compute constants
        Gx, Gy = self._cal_guidance_G(I)
        laplacian = self._generate_laplacian_matrix(h, w)
        DtG = self._divergence(Gx, Gy) # D^T * g part of Eq 14

        # Initialization
        L = I.copy()
        R = np.ones_like(I)
        N = np.zeros_like(I)
        T_x = np.zeros_like(I)
        T_y = np.zeros_like(I)
        Z_x = np.zeros_like(I)
        Z_y = np.zeros_like(I)
        
        mu = 1.0
        rho = 1.5
        
        print("Starting ADM Optimization...")
        
        # Optimization Loop (Algorithm 1)
        for k in range(self.iterations):
            # Update R (Eq 14)
            diag_R = L**2
            rhs_R = L * (I - N) + self.omega * DtG
            R = self._solve_subproblem(diag_R, laplacian, rhs_R, self.omega)
            
            # Update L (Eq 16)
            diag_L = 2 * (R**2)
            div_term = self._divergence(T_x - Z_x/mu, T_y - Z_y/mu)
            rhs_L = 2 * R * (I - N) + mu * div_term
            L = self._solve_subproblem(diag_L, laplacian, rhs_L, mu)
            
            # Update N (Eq. 18)
            N = (I - R * L) / (1 + self.delta)
            
            # Update T (Eq. 20)
            grad_L_x, grad_L_y = self._get_gradients(L)
            thresh = self.beta / mu
            T_x = self._shrinkage(grad_L_x + Z_x/mu, thresh)
            T_y = self._shrinkage(grad_L_y + Z_y/mu, thresh)
            
            # Update Z and mu
            Z_x = Z_x + mu * (grad_L_x - T_x)
            Z_y = Z_y + mu * (grad_L_y - T_y)
            mu = mu * rho

            print(f"Iteration {k+1}/{self.iterations} complete.")

        # Illumination Adjustment
        L = np.clip(L, 0, 1)
        L_adjusted = np.power(L, 1/self.gamma)
        
        # Recompose
        R = np.clip(R, 0, 1)
        I_enhanced = np.clip(R * L_adjusted, 0, 1)
        
        # Merge back to HSV and convert to RGB
        img_hsv[:, :, 2] = I_enhanced
        result_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        return np.clip(result_bgr * 255, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    dir_root = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/data"
    path_img = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/data/933_cheeseburger/og/20251119_211205.jpg"
    path_img_out = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/augmented/cheeseburger_20251119_211205_robustretinex_clipped.jpg"
    dir_out = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/img_enhanced"

    enhancer = RobustRetinex(beta=0.01, omega=0.01, delta=10.)
    result = enhancer.enhance(path_img)
    cv2.imwrite(path_img_out, result)


    # for root, dirs, files in os.walk(dir_root):
    #     for file in files:
    #         if file.endswith(".jpg") and root.find("low_light") != -1:
    #             img_path = os.path.join(root, file)
    #             out_path = os.path.join(root.replace("low_light", "restored_classic"), file)
    #             result = enhancer.enhance(img_path)
    #             cv2.imwrite(out_path, result)
    #             out_path = os.path.join(root.replace("data", "img_enhanced").replace("low_light", ""), file)
    #             cv2.imwrite(out_path, result)
