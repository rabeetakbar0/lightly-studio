<script lang="ts">
    import type { TrainConfig, TrainingOptionsView } from '$lib/api/lightly_studio_local';
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';
    import { Select } from '$lib/components/Select';
    import { presetFor } from './trainFields';

    interface Props {
        options: TrainingOptionsView;
        datasetNames: string[];
        config: TrainConfig;
        name?: string;
        dataset?: string | undefined;
        /** Show only the model size preset (for the Model tab). */
        presetOnly?: boolean;
    }
    let {
        options,
        datasetNames,
        config = $bindable(),
        name = $bindable(''),
        dataset = $bindable(undefined),
        presetOnly = false
    }: Props = $props();

    const NONE = '__none__';
    const preset = $derived(presetFor(config, options.presets));
    const presetItems = $derived([
        ...Object.keys(options.presets).map((p) => ({ value: p, label: `YOLOX-${p}` })),
        { value: 'custom', label: 'custom (edit depth and width)', disabled: true }
    ]);
    const baselineItems = $derived([
        ...options.baselines.map((b) => ({
            value: b.value,
            label: b.available ? b.label : `${b.label} (no checkpoint)`,
            group: b.group,
            disabled: !b.available
        })),
        { value: NONE, label: 'none (random initialisation)', group: 'Scratch' }
    ]);
    const applyPreset = (value: string) => {
        const [depth, width] = options.presets[value] ?? [];
        if (depth !== undefined) config = { ...config, depth, width };
    };
    // A baseline fixes the architecture, so its size is applied to depth and width.
    const applyBaseline = (value: string) => {
        const baseline = options.baselines.find((b) => b.value === value);
        const size =
            baseline?.depth != null && baseline.width != null
                ? { depth: baseline.depth, width: baseline.width }
                : {};
        config = { ...config, ...size, baseline: value === NONE ? null : value };
    };
</script>

<div class="mb-4 grid gap-4 sm:grid-cols-2">
    {#if presetOnly}
        <div class="grid gap-1.5">
            <Label>Model size</Label>
            <Select
                items={presetItems}
                value={preset}
                onValueChange={applyPreset}
                class="w-full"
                testId="train-preset"
            />
            <p class="text-xs text-muted-foreground">
                Sets depth and width below. Must match the baseline's size.
            </p>
        </div>
    {:else}
        <div class="grid gap-1.5">
            <Label>Dataset</Label>
            <Select
                items={datasetNames.map((n) => ({ value: n, label: n }))}
                value={dataset}
                placeholder="Select a dataset"
                onValueChange={(v) => (dataset = v)}
                class="w-full"
                testId="train-dataset"
            />
            <p class="text-xs text-muted-foreground">
                A prepared COCO dataset; its classes become the model's classes.
            </p>
        </div>
        <div class="grid gap-1.5">
            <Label for="train-name">Run and model name</Label>
            <Input
                id="train-name"
                bind:value={name}
                placeholder="e.g. yolox_m_slm_v5"
                data-testid="train-name"
            />
            <p class="text-xs text-muted-foreground">
                Folder under runs/ and the ONNX file name in the models folder.
            </p>
        </div>
        <div class="grid gap-1.5 sm:col-span-2">
            <Label>Baseline</Label>
            <Select
                items={baselineItems}
                value={config.baseline ?? NONE}
                onValueChange={applyBaseline}
                class="w-full"
                testId="train-baseline"
            />
            <p class="text-xs text-muted-foreground">
                A model from the models folder (needs its .pth checkpoint next to the ONNX), stock
                COCO weights from the weights folder, or nothing. Picking one sets the model size.
            </p>
        </div>
    {/if}
</div>
