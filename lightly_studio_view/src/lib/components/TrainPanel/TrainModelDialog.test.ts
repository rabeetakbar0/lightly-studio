import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import TrainModelDialog from './TrainModelDialog.svelte';
import type { TrainingOptionsView } from '$lib/api/lightly_studio_local';

const defaults = {
    depth: 0.67,
    width: 0.75,
    act: 'silu',
    input_size: 640,
    test_size: 640,
    multiscale_range: 5,
    max_labels: 50,
    max_epoch: 150,
    warmup_epochs: 5,
    no_aug_epochs: 30,
    basic_lr_per_img: 0.00015625,
    warmup_lr: 0,
    min_lr_ratio: 0.05,
    scheduler: 'yoloxwarmcos',
    ema: true,
    weight_decay: 0.0005,
    momentum: 0.9,
    seed: null,
    mosaic_prob: 1,
    mixup_prob: 1,
    enable_mixup: true,
    hsv_prob: 1,
    flip_prob: 0.5,
    degrees: 10,
    translate: 0.1,
    shear: 2,
    mosaic_scale_min: 0.1,
    mosaic_scale_max: 2,
    mixup_scale_min: 0.5,
    mixup_scale_max: 1.5,
    eval_interval: 1,
    print_interval: 20,
    save_history_ckpt: false,
    test_conf: 0.01,
    nmsthre: 0.65,
    batch_size: 16,
    devices: 1,
    data_num_workers: 4,
    fp16: true,
    occupy: false,
    cache: 'none',
    baseline: 'weights:yolox_m.pth',
    gpu_memory_cap_gb: 15,
    export_onnx: true,
    onnx_opset: 11
} as const;

const options = (ready = true): TrainingOptionsView => ({
    defaults: { ...defaults },
    presets: { s: [0.33, 0.5], m: [0.67, 0.75] },
    baselines: [
        {
            value: 'model:yolox_m_slm_v4',
            label: 'yolox_m_slm_v4',
            group: 'Trained models',
            depth: 0.67,
            width: 0.75,
            classes: ['GateArm', 'LedLight'],
            available: true
        },
        {
            value: 'weights:yolox_m.pth',
            label: 'yolox_m.pth',
            group: 'Stock YOLOX',
            depth: 0.67,
            width: 0.75,
            classes: [],
            available: true
        }
    ],
    environment: {
        yolox_dir: '/yolox',
        python: '/venv/bin/python',
        weights: ['yolox_m.pth'],
        problems: ready ? [] : ['LIGHTLY_STUDIO_YOLOX_DIR is not set.'],
        ready
    }
});

const defaultProps = {
    options: options(),
    datasetNames: ['data1', 'data2'],
    isPending: false,
    onConfirm: vi.fn(),
    onCancel: vi.fn()
};

describe('TrainModelDialog', () => {
    it('submits the name, the first dataset and the edited settings', async () => {
        const onConfirm = vi.fn();
        render(TrainModelDialog, { props: { ...defaultProps, onConfirm } });
        expect(screen.getByTestId('train-confirm')).toBeDisabled();

        await fireEvent.input(screen.getByTestId('train-name'), {
            target: { value: 'yolox_m_v5' }
        });
        await fireEvent.input(screen.getByTestId('train-batch_size'), { target: { value: '8' } });
        await fireEvent.input(screen.getByTestId('train-gpu_memory_cap_gb'), {
            target: { value: '' }
        });
        await fireEvent.click(screen.getByTestId('train-fp16'));

        await fireEvent.click(screen.getByTestId('train-tab-schedule'));
        await fireEvent.input(screen.getByTestId('train-max_epoch'), { target: { value: '3' } });

        expect(screen.getByTestId('train-confirm')).toBeEnabled();
        await fireEvent.click(screen.getByTestId('train-confirm'));
        expect(onConfirm).toHaveBeenCalledWith({
            name: 'yolox_m_v5',
            dataset: 'data1',
            config: {
                ...defaults,
                batch_size: 8,
                gpu_memory_cap_gb: null,
                fp16: false,
                max_epoch: 3
            }
        });
    });

    it('starts from the given settings', async () => {
        const onConfirm = vi.fn();
        render(TrainModelDialog, {
            props: {
                ...defaultProps,
                onConfirm,
                initialConfig: { baseline: 'model:yolox_m_slm_v4', max_epoch: 60, depth: 0.33 }
            }
        });
        await fireEvent.input(screen.getByTestId('train-name'), { target: { value: 'v5' } });
        await fireEvent.click(screen.getByTestId('train-confirm'));
        expect(onConfirm).toHaveBeenCalledWith({
            name: 'v5',
            dataset: 'data1',
            config: { ...defaults, baseline: 'model:yolox_m_slm_v4', max_epoch: 60, depth: 0.33 }
        });
    });

    it('shows the environment problems and blocks the start', async () => {
        render(TrainModelDialog, { props: { ...defaultProps, options: options(false) } });
        expect(screen.getByTestId('train-problems')).toHaveTextContent(
            'LIGHTLY_STUDIO_YOLOX_DIR is not set.'
        );
        await fireEvent.input(screen.getByTestId('train-name'), { target: { value: 'ok' } });
        expect(screen.getByTestId('train-confirm')).toBeDisabled();
    });
});
