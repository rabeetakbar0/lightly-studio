<script lang="ts">
    import { onMount } from 'svelte';
    import { toast } from 'svelte-sonner';
    import { ImagePlus } from '@lucide/svelte';
    import { Button } from '$lib/components';
    import { useMediaUpload } from '$lib/hooks/useMediaUpload/useMediaUpload.svelte';
    import {
        splitMediaFiles,
        type FrameSettings
    } from '$lib/hooks/useMediaUpload/useMediaUpload.helpers';
    import VideoFrameSettingsDialog from './VideoFrameSettingsDialog.svelte';
    import ImageUploadDialog from './ImageUploadDialog.svelte';

    interface Props {
        collectionId: string;
        /** True when the collection has no images yet; shows a standing drop target. */
        isEmpty?: boolean;
    }
    let { collectionId, isEmpty = false }: Props = $props();
    let fileInput = $state<HTMLInputElement | null>(null);

    const upload = useMediaUpload(() => ({ collectionId }));
    // Counts nested enter/leave pairs, so the overlay survives moving over child elements.
    let dragDepth = $state(0);
    let pendingVideos = $state<File[]>([]);
    let pendingImages = $state<File[]>([]);

    const carriesFiles = (event: DragEvent) =>
        Array.from(event.dataTransfer?.types ?? []).includes('Files');
    // The search box has its own image drop target; leave those drops to it.
    const isSearchDrop = (event: DragEvent) =>
        (event.target as Element | null)?.closest?.('[data-grid-search-drop-target]') != null;

    const onDragEnter = (event: DragEvent) => {
        if (!carriesFiles(event)) return;
        event.preventDefault();
        dragDepth += 1;
    };
    const onDragOver = (event: DragEvent) => {
        if (carriesFiles(event)) event.preventDefault();
    };
    const onDragLeave = (event: DragEvent) => {
        if (carriesFiles(event)) dragDepth = Math.max(0, dragDepth - 1);
    };
    const addFiles = (files: File[]) => {
        const { images, videos, unsupported } = splitMediaFiles(files);
        if (unsupported.length > 0) {
            toast.warning(`Skipped ${unsupported.length} unsupported files.`);
        }
        if (images.length > 0) pendingImages = images;
        if (videos.length > 0) pendingVideos = videos;
    };
    const confirmImages = (tag: string) => {
        upload.uploadImages(pendingImages, tag);
        pendingImages = [];
    };
    const onDrop = (event: DragEvent) => {
        dragDepth = 0;
        if (!carriesFiles(event) || isSearchDrop(event)) return;
        event.preventDefault();
        addFiles(Array.from(event.dataTransfer?.files ?? []));
    };
    const onFilesChosen = (event: Event) => {
        const input = event.currentTarget as HTMLInputElement;
        addFiles(Array.from(input.files ?? []));
        input.value = '';
    };

    const confirmVideos = (settings: FrameSettings, tags: string[]) => {
        upload.uploadVideos(pendingVideos, settings, tags);
        pendingVideos = [];
    };

    onMount(() => {
        window.addEventListener('dragenter', onDragEnter);
        window.addEventListener('dragover', onDragOver);
        window.addEventListener('dragleave', onDragLeave);
        window.addEventListener('drop', onDrop);
        return () => {
            window.removeEventListener('dragenter', onDragEnter);
            window.removeEventListener('dragover', onDragOver);
            window.removeEventListener('dragleave', onDragLeave);
            window.removeEventListener('drop', onDrop);
        };
    });
</script>

{#if dragDepth > 0}
    <div
        class="pointer-events-none absolute inset-0 z-40 flex flex-col items-center justify-center gap-2 rounded-[1vw] border-2 border-dashed border-primary bg-background/80"
        data-testid="media-drop-overlay"
    >
        <ImagePlus class="size-10 text-primary" />
        <p class="text-lg font-medium">Drop images or videos to add them to this collection</p>
        <p class="text-sm text-muted-foreground">Videos are turned into frames first.</p>
    </div>
{:else if isEmpty}
    <div
        class="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 rounded-[1vw] border-2 border-dashed border-border-hard bg-card"
        data-testid="media-drop-empty"
    >
        <ImagePlus class="size-12 text-muted-foreground" />
        <p class="text-lg font-medium">This collection is empty</p>
        <p class="max-w-md text-center text-sm text-muted-foreground">
            Drop images or videos anywhere on this page. Images are added as they are. For a video
            you choose how many frames to extract, and those frames are added as images.
        </p>
        <Button
            variant="outline"
            buttonProps={{ type: 'button', onclick: () => fileInput?.click() }}
        >
            Choose files
        </Button>
        <input
            bind:this={fileInput}
            type="file"
            multiple
            accept="image/*,video/*,.mkv,.avi,.mov,.mp4,.webm"
            class="hidden"
            onchange={onFilesChosen}
            data-testid="media-file-input"
        />
    </div>
{/if}

{#if pendingImages.length > 0}
    <ImageUploadDialog
        images={pendingImages}
        onConfirm={confirmImages}
        onCancel={() => (pendingImages = [])}
    />
{/if}

{#if pendingVideos.length > 0}
    <VideoFrameSettingsDialog
        videos={pendingVideos}
        planFrames={upload.planFrames}
        onConfirm={confirmVideos}
        onCancel={() => (pendingVideos = [])}
    />
{/if}
