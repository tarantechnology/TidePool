"""Conservative, optional target reduction from an existing separator estimate.

Uses the original mixture's phase through mask_refine. A target-derived activity
envelope limits processing when the estimated target is weak. Neither the ratio
mask nor the activity envelope is a calibrated confidence or speech detector.
No clean references or additional model calls are used.
"""
import numpy as np

from .refine import mask_refine, stft


def suppress_target(mixture, target, *, strength=.9, mask_threshold=.85,
                    activity_low_db=-36., activity_high_db=-24., n_fft=2048):
    """Return an edited (channels, samples) mixture; strength 0 leaves it intact.

    strength=.9 retains 10% amplitude in fully target-dominated regions. Speech
    sharing frequencies with the target can still be affected. Preserve the raw
    estimate separately; this is a listening candidate, not a guaranteed repair.
    """
    for name, value in [('strength', strength), ('mask_threshold', mask_threshold),
                        ('activity_low_db', activity_low_db), ('activity_high_db', activity_high_db)]:
        if not np.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if not 0 <= strength <= 1 or not 0 <= mask_threshold < 1:
        raise ValueError('strength must be in [0,1] and mask_threshold in [0,1)')
    if not activity_low_db < activity_high_db <= 0:
        raise ValueError('activity levels must be ordered at or below 0 dB')
    if n_fft not in (512, 1024, 2048, 4096):
        raise ValueError('unsupported FFT size')
    mixture, target = np.asarray(mixture, np.float32), np.asarray(target, np.float32)
    if mixture.ndim != 2 or mixture.shape != target.shape or not mixture.shape[0]:
        raise ValueError('mixture and target must have identical (channels, samples) shapes')
    if not np.isfinite(mixture).all() or not np.isfinite(target).all():
        raise ValueError('audio must be finite')
    if mixture.shape[-1] < 2 or strength == 0 or not np.any(target):
        return mixture.copy()

    # 1024-sample activity window, centered every 256 samples (5.33 ms at 48 kHz).
    power = np.mean(np.abs(stft(target.mean(axis=0), 1024))**2, axis=1)
    level = 10*np.log10(np.maximum(power, 1e-20))
    relative = level-level.max()
    activity = np.clip((relative-activity_low_db)/(activity_high_db-activity_low_db), 0., 1.)
    activity = activity*activity*(3-2*activity)
    # Smooth activity transitions without using source-reference timing.
    weights = np.array([1., 2., 3., 2., 1.])/9
    activity = np.convolve(np.pad(activity, (2, 2), mode='edge'), weights, mode='valid')
    activity = np.interp(np.arange(mixture.shape[-1]), np.arange(len(activity))*256, activity)
    removed = mask_refine(mixture, target, p=2., n_fft=n_fft, threshold=mask_threshold)
    return (mixture-strength*removed*activity[None, :]).astype(np.float32)
