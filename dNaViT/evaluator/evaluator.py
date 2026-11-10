# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""This file contains a class to evalute the reconstruction results."""
import warnings

from typing import Sequence, Optional, Mapping, Text
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F

from .inception import get_inception_model
from modeling.modules import LPIPS

def uniform(kernel_size: int) -> torch.Tensor:
    """Computes 1D uniform kernel.

    Args:
        kernel_size: An integer, size of the 1D uniform kernel.
    
    Returns:
        A torch.Tensor which is 1D uniform kernel of size `kernel_size`.
    """ 
    max_, min_ = 2.5, -2.5
    ksize_half = (kernel_size - 1) * 0.5
    kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    for i, j in enumerate(kernel):
        if min_ <= j <= max_:
            kernel[i] = 1 / (max_ - min_)
        else:
            kernel[i] = 0

    return kernel

def gaussian(kernel_size: int, sigma: float) -> torch.Tensor:
    """Computes 1D Gaussian kernel.

    Args:
        kernel_size: An integer, size of the 1D Gaussian kernel.
        sigma: A float, standard deviation of the 1D Gaussian kernel.
    
    Returns:
        A torch.Tensor, which is 1D Gaussian kernel of size `kernel_size`.
    """
    ksize_half = (kernel_size - 1) * 0.5
    kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
    return gauss / gauss.sum()

def gaussian_or_uniform_kernel(kernel_size: Sequence[int], sigma: Sequence[float], use_gaussian: bool) -> torch.Tensor:
    """Computes 2D Gaussian or unniform kernel.

    Args:
        kernel_size: A sequence of integer, size of the 2D kernel.
        sigma: A sequence of float, standard deviation of the 2D kernel.
        use_gaussian: A boolea, if True, uses Gaussian kernel, otherwise uniform kernel.
    
    Returns:
        A torch.Tensor, which is 2D Gaussian or uniform kernel.
    
    Raises:
        ValueError: If kernel_size or sigma is not 2-dimensional.
    """
    if len(kernel_size)!= 2 or len(sigma)!= 2:
        raise ValueError("Kernel size and sigma must be 2-dimensional.")
    
    if use_gaussian:
        kernel_x = gaussian(kernel_size[0], sigma[0])
        kernel_y = gaussian(kernel_size[1], sigma[1])
    else:
        kernel_x = uniform(kernel_size[0])
        kernel_y = uniform(kernel_size[1])

    return torch.outer(kernel_x, kernel_y)

def get_covariance(sigma: torch.Tensor, total: torch.Tensor, num_examples: int) -> torch.Tensor:
    """Computes covariance of the input tensor.

    Args:
        sigma: A torch.Tensor, sum of outer products of input features.
        total: A torch.Tensor, sum of all input features.
        num_examples: An integer, number of examples in the input tensor.
    Returns:
        A torch.Tensor, covariance of the input tensor.
    """
    if num_examples == 0:
        return torch.zeros_like(sigma)

    sub_matrix = torch.outer(total, total)
    sub_matrix = sub_matrix / num_examples

    return (sigma - sub_matrix) / (num_examples - 1)

class ReconstructionEvaluator:
    def __init__(
        self,
        device,
        enable_rfid: bool = True,
        enable_inception_score: bool = True,
        enable_psnr_score: bool = True,
        enable_ssim_score: bool = True,
        enable_lpips_score: bool = True,
        enable_mse_error: bool = True,
        enable_mae_error: bool = True,
    ):
        """Initializes the reconstruction evaluator.

        Args:
            device: The device to use for evaluation.
            enable_rfid: A boolean, whether enabling rFID score.
            enable_inception_score: A boolean, whether enabling Inception Score.
            enable_psnr_score: A boolean, whether enabling PSNR Score.
            enable_ssim_score: A boolean, whether enabling SSIM Score.
            enable_lpips_score: A boolean, whether enabling LPIPS Score.
            enable_mse_error: A boolean, whether enabling MSE Error.
            enable_mae_error: A boolean, whether enabling MAE Error.
        """
        self._device = device

        self._enable_rfid = enable_rfid
        self._enable_inception_score = enable_inception_score
        self._enable_psnr_score = enable_psnr_score
        self._enable_ssim_score = enable_ssim_score
        self._enable_lpips_score = enable_lpips_score
        self._enable_mse_error = enable_mse_error
        self._enable_mae_error = enable_mae_error

        # Variables related to PSNR. We assume all data to be in range 0.0 - 1.0.
        self._data_range = 1.0

        # Variables related to SSIM.
        ssim_kernel_size = (11, 11)
        ssim_sigma = (1.5, 1.5)
        ssim_k1 = 0.01
        ssim_k2 = 0.03
        ssim_gaussian = True
        self._ssim_c1 = (ssim_k1 * self._data_range) ** 2
        self._ssim_c2 = (ssim_k2 * self._data_range) ** 2
        self._ssim_pad_h = (ssim_kernel_size[0] - 1) // 2
        self._ssim_pad_w = (ssim_kernel_size[1] - 1) // 2
        self._ssim_kernel = gaussian_or_uniform_kernel(
            kernel_size=ssim_kernel_size,
            sigma=ssim_sigma,
            use_gaussian=ssim_gaussian
        ).to(self._device)

        # Variables related to Inception score and rFID.
        self._inception_model = None
        self._is_num_features = 0
        self._rfid_num_features = 0
        if self._enable_inception_score or self._enable_rfid:
            self._rfid_num_features = 2048
            self._is_num_features = 1008
            self._inception_model = get_inception_model().to(self._device)
            self._inception_model.eval()
        self._is_eps = 1e-16
        self._rfid_eps = 1e-6

        self._lpips_model = None
        if self._enable_lpips_score:
            self._lpips_model = LPIPS().to(self._device)
            self._lpips_model.eval()

        self.reset_metrics()

    def reset_metrics(self):
        """Resets all metrics."""
        self._num_examples = 0
        self._num_updates = 0
        self._sum_of_batchwise_absolute_errors = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._sum_of_batchwise_squared_errors = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._sum_of_batchwise_psnr = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._sum_of_ssim = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._is_prob_total = torch.zeros(
            self._is_num_features, dtype=torch.float64, device=self._device
        )
        self._is_total_kl_d = torch.zeros(
            self._is_num_features, dtype=torch.float64, device=self._device
        )
        self._rfid_real_sigma = torch.zeros(
            (self._rfid_num_features, self._rfid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._rfid_real_total = torch.zeros(
            self._rfid_num_features, dtype=torch.float64, device=self._device
        )
        self._rfid_fake_sigma = torch.zeros(
            (self._rfid_num_features, self._rfid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._rfid_fake_total = torch.zeros(
            self._rfid_num_features, dtype=torch.float64, device=self._device
        )
        self._sum_of_lpips = torch.tensor(0.0, dtype=torch.float64, device=self._device)

    def update(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        codebook_indices: Optional[torch.Tensor] = None
    ):
        """Updates the metrics with the given images.

        Args:
            real_images: A torch.Tensor, the real images.
            fake_images: A torch.Tensor, the fake images.
            codebook_indices: A torch.Tensor, unused (kept for API compatibility).

        Raises:
            ValueError: If the fake images is not in RGB (3 channel).
            ValueError: If the fake and real images have different shape.
        """

        batch_size = real_images.shape[0]
        dim = tuple(range(1, real_images.ndim))
        self._num_examples += batch_size
        self._num_updates += 1

        if self._enable_mae_error:
            absolute_errors = torch.abs(fake_images - real_images.view_as(fake_images))
            self._sum_of_batchwise_absolute_errors += torch.mean(absolute_errors, dim=dim).sum().to(self._device)

        if self._enable_mse_error:
            squared_errors = torch.pow(fake_images - real_images.view_as(fake_images), 2)
            self._sum_of_batchwise_squared_errors += torch.mean(squared_errors, dim=dim).sum().to(self._device)

        if self._enable_psnr_score:
            mse_error = torch.pow(fake_images.double() - real_images.view_as(fake_images).double(), 2).mean(dim=dim)
            self._sum_of_batchwise_psnr += torch.sum(10.0 * torch.log10(self._data_range ** 2 / (mse_error + 1e-10))).to(
                device=self._device
            )
        
        if self._enable_ssim_score:
            channel = fake_images.size(1)
            if channel != 3:
                raise ValueError(f"Currently only tested for rgb, got channel {channel}")

            if len(self._ssim_kernel.shape) < 4:
                self._ssim_kernel = self._ssim_kernel.expand(channel, 1, -1, -1)

            ssim_real_images = torch.clone(real_images)
            ssim_fake_images = torch.clone(fake_images)

            ssim_fake_images = F.pad(ssim_fake_images,
             [self._ssim_pad_w, self._ssim_pad_w, self._ssim_pad_h, self._ssim_pad_h],
              mode="reflect")
            ssim_real_images = F.pad(ssim_real_images,
             [self._ssim_pad_w, self._ssim_pad_w, self._ssim_pad_h, self._ssim_pad_h],
              mode="reflect")

            input_list = [
                ssim_fake_images,
                ssim_real_images,
                torch.pow(ssim_fake_images, 2),
                torch.pow(ssim_real_images, 2),
                ssim_fake_images * ssim_real_images
            ]
            # Depthwise conv
            outputs = F.conv2d(torch.cat(input_list), self._ssim_kernel, groups=channel)
            output_list = [outputs[x * batch_size : (x + 1) * batch_size] for x in range(len(input_list))]

            mu_pred_sq = output_list[0].pow(2)
            mu_target_sq = output_list[1].pow(2)
            mu_pred_target = output_list[0] * output_list[1]

            sigma_pred_sq = output_list[2] - mu_pred_sq
            sigma_target_sq = output_list[3] - mu_target_sq
            sigma_pred_target = output_list[4] - mu_pred_target

            a1 = 2 * mu_pred_target + self._ssim_c1
            a2 = 2 * sigma_pred_target + self._ssim_c2
            b1 = mu_pred_sq + mu_target_sq + self._ssim_c1
            b2 = sigma_pred_sq + sigma_target_sq + self._ssim_c2

            ssim_idx = (a1 * a2) / (b1 * b2)
            self._sum_of_ssim += torch.mean(ssim_idx, (1, 2, 3), dtype=torch.float64).sum().to(self._device)

        if self._enable_inception_score or self._enable_rfid:
            # Quantize to uint8 as a real image.
            fake_inception_images = (fake_images * 255).to(torch.uint8)
            features_fake = self._inception_model(fake_inception_images)
            inception_logits_fake = features_fake["logits_unbiased"]
            inception_probabilities_fake = F.softmax(inception_logits_fake, dim=-1)
        
        if self._enable_inception_score:
            probabiliies_sum = torch.sum(inception_probabilities_fake, 0, dtype=torch.float64)

            log_prob = torch.log(inception_probabilities_fake + self._is_eps)
            if log_prob.dtype != inception_probabilities_fake.dtype:
                log_prob = log_prob.to(inception_probabilities_fake)
            kl_sum = torch.sum(inception_probabilities_fake * log_prob, 0, dtype=torch.float64)

            self._is_prob_total += probabiliies_sum
            self._is_total_kl_d += kl_sum

        if self._enable_rfid:
            real_inception_images = (real_images * 255).to(torch.uint8)
            features_real = self._inception_model(real_inception_images)
            if (features_real['2048'].shape[0] != features_fake['2048'].shape[0] or
                features_real['2048'].shape[1] != features_fake['2048'].shape[1]):
                raise ValueError(f"Number of features should be equal for real and fake.")

            for f_real, f_fake in zip(features_real['2048'], features_fake['2048']):
                self._rfid_real_total += f_real
                self._rfid_fake_total += f_fake

                self._rfid_real_sigma += torch.outer(f_real, f_real)
                self._rfid_fake_sigma += torch.outer(f_fake, f_fake)

        if self._enable_lpips_score:
            # from [0, 1] to [-1. 1]
            lpips = self._lpips_model(real_images * 2. - 1., fake_images * 2. - 1.)
            self._sum_of_lpips += lpips.sum()


    def result(self) -> Mapping[Text, torch.Tensor]:
        """Returns the evaluation result."""
        eval_score = {}

        if self._num_examples < 1:
            raise ValueError("No examples to evaluate.")

        if self._enable_mae_error:
            mae_error = self._sum_of_batchwise_absolute_errors.item() / self._num_examples
            eval_score["MAE"] = mae_error

        if self._enable_mse_error:
            mse_error = self._sum_of_batchwise_squared_errors.item() / self._num_examples
            eval_score["MSE"] = mse_error

        if self._enable_psnr_score:
            psnr_score = self._sum_of_batchwise_psnr.item() / self._num_examples
            eval_score["PSNR"] = psnr_score
        
        if self._enable_ssim_score:
            ssim_score = self._sum_of_ssim.item() / self._num_examples
            eval_score["SSIM"] = ssim_score
        
        if self._enable_inception_score:
            mean_probs = self._is_prob_total / self._num_examples
            log_mean_probs = torch.log(mean_probs + self._is_eps)
            if log_mean_probs.dtype != self._is_prob_total.dtype:
                log_mean_probs = log_mean_probs.to(self._is_prob_total)
            excess_entropy = self._is_prob_total * log_mean_probs
            avg_kl_d = torch.sum(self._is_total_kl_d - excess_entropy) / self._num_examples

            inception_score = torch.exp(avg_kl_d).item()
            eval_score["InceptionScore"] = inception_score

        if self._enable_rfid:
            mu_real = self._rfid_real_total / self._num_examples
            mu_fake = self._rfid_fake_total / self._num_examples
            sigma_real = get_covariance(self._rfid_real_sigma, self._rfid_real_total, self._num_examples)
            sigma_fake = get_covariance(self._rfid_fake_sigma, self._rfid_fake_total, self._num_examples)

            mu_real, mu_fake = mu_real.cpu(), mu_fake.cpu()
            sigma_real, sigma_fake = sigma_real.cpu(), sigma_fake.cpu()

            diff = mu_real - mu_fake

            # Product might be almost singular.
            covmean, _ = linalg.sqrtm(sigma_real.mm(sigma_fake).numpy(), disp=False)
            # Numerical error might give slight imaginary component.
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError("Imaginary component {}".format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            if not np.isfinite(covmean).all():
                tr_covmean = np.sum(np.sqrt((
                    (np.diag(sigma_real) * self._rfid_eps) * (np.diag(sigma_fake) * self._rfid_eps))
                    / (self._rfid_eps * self._rfid_eps)
                ))

            rfid = float(diff.dot(diff).item() + torch.trace(sigma_real) + torch.trace(sigma_fake) 
                - 2 * tr_covmean
            )
            if torch.isnan(torch.tensor(rfid)) or torch.isinf(torch.tensor(rfid)):
                warnings.warn("The product of covariance of train and test features is out of bounds.")

            eval_score["rFID"] = rfid

        if self._enable_lpips_score:
            lpips = self._sum_of_lpips.item() / self._num_examples
            eval_score["LPIPS"] = lpips

        return eval_score
