import type { TrainConfig } from '$lib/api/lightly_studio_local';

/** One editable setting of the train dialog. */
export interface TrainField {
    key: keyof TrainConfig;
    label: string;
    hint: string;
    kind: 'number' | 'nullable-number' | 'boolean' | 'select';
    step?: number;
    min?: number;
    options?: string[];
}

export interface TrainFieldGroup {
    id: string;
    label: string;
    fields: TrainField[];
}

const num = (
    key: keyof TrainConfig,
    label: string,
    hint: string,
    step = 1,
    min = 0
): TrainField => ({ key, label, hint, kind: 'number', step, min });
const bool = (key: keyof TrainConfig, label: string, hint: string): TrainField => ({
    key,
    label,
    hint,
    kind: 'boolean'
});

/** The parameter groups shown as tabs; defaults come from the backend. */
export const TRAIN_FIELD_GROUPS: TrainFieldGroup[] = [
    {
        id: 'run',
        label: 'Run',
        fields: [
            num('batch_size', 'Batch size', 'Images per step over all GPUs (-b).', 1, 1),
            num('devices', 'GPUs', 'Number of GPUs (-d).', 1, 1),
            bool('fp16', 'Mixed precision', 'Train with fp16 (--fp16); halves memory.'),
            bool('occupy', 'Occupy GPU memory first', 'Grab the memory up front (-o).'),
            {
                key: 'cache',
                label: 'Image cache',
                hint: 'Cache decoded images in RAM or on disk for faster epochs.',
                kind: 'select',
                options: ['none', 'ram', 'disk']
            },
            num('data_num_workers', 'Data loader workers', 'Lower it when RAM is short.', 1, 0),
            {
                key: 'gpu_memory_cap_gb',
                label: 'GPU memory cap (GB)',
                hint: 'Hard limit for this process; empty means no cap.',
                kind: 'nullable-number',
                step: 0.5,
                min: 1
            },
            bool('export_onnx', 'Export ONNX when done', 'Puts the model into the models folder.'),
            num('onnx_opset', 'ONNX opset', 'Opset version of the exported graph.', 1, 9)
        ]
    },
    {
        id: 'model',
        label: 'Model',
        fields: [
            num('depth', 'Depth factor', 'Set by the size preset; 0.67 is YOLOX-M.', 0.01, 0.01),
            num('width', 'Width factor', 'Set by the size preset; 0.75 is YOLOX-M.', 0.01, 0.01),
            {
                key: 'act',
                label: 'Activation',
                hint: 'silu is the YOLOX default.',
                kind: 'select',
                options: ['silu', 'relu', 'lrelu']
            },
            num('input_size', 'Train input size', 'Square, multiple of 32.', 32, 32),
            num('test_size', 'Eval and export size', 'Keep equal to the train size.', 32, 32),
            num(
                'multiscale_range',
                'Multiscale range',
                'Size ± range × 32 during training; 0 disables.',
                1,
                0
            ),
            num(
                'max_labels',
                'Max boxes per image',
                'Boxes above this are dropped per image.',
                1,
                1
            )
        ]
    },
    {
        id: 'schedule',
        label: 'Schedule',
        fields: [
            num('max_epoch', 'Epochs', 'Total training epochs.', 1, 1),
            num('warmup_epochs', 'Warm-up epochs', 'Linear warm-up at the start.', 1, 0),
            num(
                'no_aug_epochs',
                'Clean epochs at the end',
                'No mosaic/mixup, L1 box loss on.',
                1,
                0
            ),
            num(
                'basic_lr_per_img',
                'Learning rate per image',
                'lr = this × batch size.',
                0.00001,
                0
            ),
            num('warmup_lr', 'Warm-up start lr', 'Learning rate at step 0.', 0.00001, 0),
            num('min_lr_ratio', 'Min lr ratio', 'Final lr as a share of the peak.', 0.01, 0),
            {
                key: 'scheduler',
                label: 'Scheduler',
                hint: 'yoloxwarmcos is the YOLOX default.',
                kind: 'select',
                options: ['yoloxwarmcos', 'warmcos', 'cos', 'yoloxsemiwarmcos']
            },
            bool('ema', 'EMA weights', 'Keep an exponential moving average of the weights.'),
            num('weight_decay', 'Weight decay', 'SGD weight decay.', 0.0001, 0),
            num('momentum', 'Momentum', 'SGD momentum.', 0.01, 0),
            {
                key: 'seed',
                label: 'Seed',
                hint: 'Fixes the random seed; slower, but reproducible. Empty means random.',
                kind: 'nullable-number',
                step: 1,
                min: 0
            }
        ]
    },
    {
        id: 'augmentation',
        label: 'Augmentation',
        fields: [
            num(
                'mosaic_prob',
                'Mosaic probability',
                'Share of samples built as a 4-image mosaic.',
                0.05
            ),
            num(
                'mixup_prob',
                'Mixup probability',
                'Share of mosaics blended with another image.',
                0.05
            ),
            bool('enable_mixup', 'Enable mixup', 'Switch mixup off entirely.'),
            num('hsv_prob', 'HSV probability', 'Colour jitter.', 0.05),
            num('flip_prob', 'Flip probability', 'Horizontal flips.', 0.05),
            num('degrees', 'Rotation (degrees)', 'Random rotation range ± this.', 1),
            num('translate', 'Translation', 'Random shift as a share of the size.', 0.01),
            num('shear', 'Shear (degrees)', 'Random shear range ± this.', 0.5),
            num(
                'mosaic_scale_min',
                'Mosaic scale min',
                'Lower bound of the random scale.',
                0.1,
                0.01
            ),
            num(
                'mosaic_scale_max',
                'Mosaic scale max',
                'Upper bound of the random scale.',
                0.1,
                0.01
            ),
            num(
                'mixup_scale_min',
                'Mixup scale min',
                'Lower bound of the mixup image scale.',
                0.1,
                0.01
            ),
            num(
                'mixup_scale_max',
                'Mixup scale max',
                'Upper bound of the mixup image scale.',
                0.1,
                0.01
            )
        ]
    },
    {
        id: 'evaluation',
        label: 'Evaluation',
        fields: [
            num(
                'eval_interval',
                'Evaluate every N epochs',
                'The best checkpoint is picked by this eval.',
                1,
                1
            ),
            num('print_interval', 'Log every N iterations', 'Progress lines in the log.', 1, 1),
            bool(
                'save_history_ckpt',
                'Keep every epoch checkpoint',
                'Off keeps only best and latest.'
            ),
            num(
                'test_conf',
                'Eval confidence threshold',
                'Boxes below are dropped before mAP.',
                0.01
            ),
            num('nmsthre', 'Eval NMS threshold', 'IoU threshold of non-maximum suppression.', 0.05)
        ]
    }
];

export const RUN_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

/** The preset whose depth and width match the config, or "custom". */
export function presetFor(config: TrainConfig, presets: Record<string, number[]>): string {
    const match = Object.entries(presets).find(
        ([, [depth, width]]) => depth === config.depth && width === config.width
    );
    return match ? match[0] : 'custom';
}
