<script lang="ts">
    import { onMount, untrack } from 'svelte';
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Input } from '$lib/components/ui/input';
    import {
        defaultFrameSettings,
        fileStem,
        formatDuration,
        readVideoDuration,
        type FrameSettings
    } from '$lib/hooks/useMediaUpload/useMediaUpload.helpers';
    import FrameSettingsFields from './FrameSettingsFields.svelte';

    interface Props {
        videos: File[];
        planFrames: (durations: number[], settings: FrameSettings) => Promise<number>;
        onConfirm: (settings: FrameSettings, tags: string[]) => void;
        onCancel: () => void;
    }
    let { videos, planFrames, onConfirm, onCancel }: Props = $props();

    let settings = $state<FrameSettings>({ ...defaultFrameSettings });
    // One tag per video, defaulting to the video name; the frames of each video get its tag.
    // The dialog mounts once per drop, so the initial names are all it needs.
    let tags = $state<string[]>(untrack(() => videos.map(fileStem)));
    let durations = $state<number[] | null>(null);
    let totalFrames = $state<number | null>(null);

    onMount(async () => {
        const read = await Promise.allSettled(videos.map(readVideoDuration));
        durations = read.map((r) => (r.status === 'fulfilled' ? r.value : 0));
    });

    // A cleared number input binds null, which would pass plain comparisons.
    const isNumber = (value: unknown): value is number =>
        typeof value === 'number' && Number.isFinite(value);
    const isValid = $derived(
        isNumber(settings.edgeWindowSeconds) &&
            settings.edgeWindowSeconds >= 0 &&
            isNumber(settings.edgeIntervalSeconds) &&
            settings.edgeIntervalSeconds > 0 &&
            isNumber(settings.middleFrames) &&
            settings.middleFrames >= 0 &&
            isNumber(settings.maxFrames) &&
            settings.maxFrames >= 1
    );

    // Only the latest plan request may set the estimate; earlier ones can resolve later.
    let planRequest = 0;
    $effect(() => {
        const snapshot = { ...settings };
        const known = durations;
        const request = ++planRequest;
        if (!known || !isValid) {
            totalFrames = null;
            return;
        }
        planFrames(known, snapshot)
            .then((total) => {
                if (request === planRequest) totalFrames = total;
            })
            .catch(() => {
                if (request === planRequest) totalFrames = null;
            });
    });

    const totalDuration = $derived(durations?.reduce((a, b) => a + b, 0) ?? 0);

    const submit = (event: SubmitEvent) => {
        event.preventDefault();
        if (isValid)
            onConfirm(
                { ...settings },
                tags.map((tag) => tag.trim())
            );
    };
</script>

<Dialog.Root bind:open={() => true, onCancel}>
    <Dialog.Content class="sm:max-w-xl">
        <form onsubmit={submit}>
            <Dialog.Header>
                <Dialog.Title
                    >Extract frames from {videos.length === 1 ? 'video' : 'videos'}</Dialog.Title
                >
                <Dialog.Description>
                    The frames are added to this collection as images. The start and the end of each
                    video are sampled densely, the middle sparsely.
                </Dialog.Description>
            </Dialog.Header>
            <div class="grid gap-4 py-4">
                <ul class="grid max-h-40 gap-2 overflow-y-auto text-sm" data-testid="video-list">
                    {#each videos as video, i (video.name + i)}
                        <li class="grid grid-cols-[1fr_auto_minmax(8rem,12rem)] items-center gap-3">
                            <span class="truncate" title={video.name}>{video.name}</span>
                            <span class="shrink-0 text-muted-foreground">
                                {durations ? formatDuration(durations[i]) : '…'}
                            </span>
                            <Input
                                bind:value={tags[i]}
                                placeholder="Tag for its frames (optional)"
                                aria-label={`Tag for frames of ${video.name}`}
                                data-testid="video-tag-input"
                            />
                        </li>
                    {/each}
                </ul>
                <FrameSettingsFields bind:settings />
                <p class="text-sm" data-testid="frame-plan">
                    {#if totalFrames !== null}
                        About <strong>{totalFrames}</strong> frames will be added from
                        {formatDuration(totalDuration)} of video.
                    {:else if !isValid}
                        Every field needs a number: interval above 0, maximum frames at least 1.
                    {:else}
                        Estimating the number of frames…
                    {/if}
                </p>
            </div>
            <Dialog.Footer>
                <Button variant="outline" buttonProps={{ type: 'button', onclick: onCancel }}>
                    Cancel
                </Button>
                <Button buttonProps={{ type: 'submit', disabled: !isValid }}>Add frames</Button>
            </Dialog.Footer>
        </form>
    </Dialog.Content>
</Dialog.Root>
