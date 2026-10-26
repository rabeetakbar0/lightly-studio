<script lang="ts">
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';
    import type { FrameSettings } from '$lib/hooks/useMediaUpload/useMediaUpload.helpers';

    interface Props {
        settings: FrameSettings;
    }
    let { settings = $bindable() }: Props = $props();

    const fields: { key: keyof FrameSettings; label: string; hint: string; step: number }[] = [
        {
            key: 'edgeWindowSeconds',
            label: 'Dense window at start and end (s)',
            hint: 'Seconds at each end of the video that are sampled densely.',
            step: 1
        },
        {
            key: 'edgeIntervalSeconds',
            label: 'Interval inside the dense windows (s)',
            hint: 'One frame every this many seconds, at the start and at the end.',
            step: 0.1
        },
        {
            key: 'middleFrames',
            label: 'Frames from the middle',
            hint: 'Evenly spaced over the rest of the video, whatever its length.',
            step: 1
        },
        {
            key: 'maxFrames',
            label: 'Maximum frames per video',
            hint: 'The selection is thinned evenly above this count.',
            step: 1
        }
    ];
</script>

<div class="grid gap-4 sm:grid-cols-2">
    {#each fields as field (field.key)}
        <div class="grid gap-1.5">
            <Label for={`frame-${field.key}`}>{field.label}</Label>
            <Input
                id={`frame-${field.key}`}
                type="number"
                min={0}
                step={field.step}
                bind:value={settings[field.key]}
            />
            <p class="text-xs text-muted-foreground">{field.hint}</p>
        </div>
    {/each}
</div>
