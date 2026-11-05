<script lang="ts">
    import { Button, Spinner, Typography } from '$lib/components';
    import { Label } from '$lib/components/ui/label';
    import { Select } from '$lib/components/Select';
    import { Clapperboard, Play, Upload, X } from '@lucide/svelte';
    import { toast } from 'svelte-sonner';
    import {
        formatBytes,
        isRunningTest,
        useVideoTestActions,
        useVideoTestOptions,
        useVideoTests
    } from '$lib/hooks/useVideoTests/useVideoTests.svelte';
    import VideoTestLogDialog from './VideoTestLogDialog.svelte';
    import VideoTestRunRow from './VideoTestRunRow.svelte';

    interface Props {
        onClose: () => void;
    }
    let { onClose }: Props = $props();

    /** The extensions App/test_pipeline.py picks up from its input folder. */
    const VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov'];

    const options = useVideoTestOptions();
    const runsQuery = useVideoTests();
    const { startMutation, start, stop } = useVideoTestActions();

    const runs = $derived(runsQuery.data ?? []);
    const running = $derived(runs.some(isRunningTest));
    const ready = $derived(options.data?.ready ?? false);
    const modelItems = $derived(
        (options.data?.models ?? []).map((m) => ({
            value: m.name,
            label: `${m.name} (${m.classes.join(', ')})`
        }))
    );

    let files = $state<File[]>([]);
    let model = $state<string | undefined>(undefined);
    let dragging = $state(false);
    let fileInput = $state<HTMLInputElement | null>(null);
    let logName = $state<string | null>(null);
    const logRun = $derived(runs.find((r) => r.name === logName));

    // Default to the first model (newest) once the list is there.
    $effect(() => {
        if (model === undefined && modelItems.length > 0) model = modelItems[0].value;
    });
    // Clear the selection once the upload went through.
    $effect(() => {
        if (startMutation.isSuccess && !startMutation.isPending) files = [];
    });

    // Announce when a test that was running ends, with where its videos went.
    let previous = new Map<string, string>();
    $effect(() => {
        const now = new Map(runs.map((r) => [r.name, r.status]));
        for (const run of runs) {
            if (previous.get(run.name) !== 'running' || run.status === 'running') continue;
            const done = run.videos.filter((v) => v.status === 'done').length;
            if (run.status === 'finished')
                toast.success(
                    `Video test ${run.name} finished: ${done} processed video${done === 1 ? '' : 's'} saved in ${run.output_dir}`
                );
            else if (run.status === 'failed')
                toast.error(`Video test ${run.name} failed${run.error ? `: ${run.error}` : ''}`);
        }
        previous = now;
    });

    const isVideo = (file: File) =>
        VIDEO_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext));
    const addFiles = (added: File[]) => {
        const videos = added.filter(isVideo);
        if (videos.length < added.length)
            toast.warning(
                `Skipped ${added.length - videos.length} file(s) that are not ${VIDEO_EXTENSIONS.join(', ')} videos.`
            );
        files = [...files, ...videos];
    };
    const removeFile = (index: number) => {
        files = files.filter((_, i) => i !== index);
    };

    const carriesFiles = (event: DragEvent) =>
        Array.from(event.dataTransfer?.types ?? []).includes('Files');
    // The grid's drop zone listens on window; drops on this panel stay here.
    const onDragOver = (event: DragEvent) => {
        if (!carriesFiles(event)) return;
        event.preventDefault();
        event.stopPropagation();
        dragging = true;
    };
    const onDragLeave = (event: DragEvent) => {
        event.stopPropagation();
        if (!(event.currentTarget as Element).contains(event.relatedTarget as Node | null))
            dragging = false;
    };
    const onDrop = (event: DragEvent) => {
        if (!carriesFiles(event)) return;
        event.preventDefault();
        event.stopPropagation();
        dragging = false;
        addFiles(Array.from(event.dataTransfer?.files ?? []));
    };
    const onFilesChosen = (event: Event) => {
        const input = event.currentTarget as HTMLInputElement;
        addFiles(Array.from(input.files ?? []));
        input.value = '';
    };

    const canRun = $derived(
        ready && files.length > 0 && !!model && !running && !startMutation.isPending
    );
    const totalBytes = $derived(files.reduce((sum, f) => sum + f.size, 0));
</script>

<div
    class="flex min-h-0 flex-1 flex-col rounded-[1vw] bg-card p-4"
    data-testid="video-test-panel"
    role="region"
    aria-label="Video test"
    ondragenter={onDragOver}
    ondragover={onDragOver}
    ondragleave={onDragLeave}
    ondrop={onDrop}
>
    <div class="mb-3 flex items-center justify-between">
        <Typography variant="h5" component="h2">Video test</Typography>
        <Button
            icon={X}
            ariaLabel="Close video test panel"
            buttonProps={{ size: 'icon', onclick: onClose, class: 'h-8 w-8' }}
        />
    </div>
    <p class="mb-3 text-sm text-muted-foreground">
        Runs the TestApp gate-arm pipeline (detection and LED blink counting) on recordings with a
        model of the models folder, and writes annotated videos.
        {#if options.data?.uploads_dir}
            Videos are saved to <span class="font-mono"
                >{options.data.uploads_dir}/&lt;time&gt;</span
            >, results to <span class="font-mono">{options.data.results_dir}/&lt;time&gt;</span>.
        {/if}
    </p>

    {#if options.isLoading}
        <Spinner size="medium" align="center" />
    {:else if options.data && !ready}
        <div
            class="mb-3 rounded-md border border-red-500/50 p-3 text-sm"
            data-testid="video-test-problems"
        >
            <p class="font-medium">Video tests are not set up:</p>
            <ul class="list-disc pl-5">
                {#each options.data.problems as problem (problem)}<li>{problem}</li>{/each}
            </ul>
            <p class="mt-1 text-muted-foreground">
                Set LIGHTLY_STUDIO_TESTAPP_DIR and LIGHTLY_STUDIO_TESTAPP_PYTHON in
                lightly_studio/.env and restart the backend.
            </p>
        </div>
    {:else}
        <div
            class={`mb-3 flex flex-col items-center gap-2 rounded-md border-2 border-dashed p-4 text-center ${dragging ? 'border-primary bg-primary/5' : 'border-border'}`}
            data-testid="video-test-drop"
        >
            <Clapperboard class="size-8 text-muted-foreground" />
            <p class="text-sm">Drop one or more videos here</p>
            <Button
                variant="outline"
                icon={Upload}
                buttonProps={{ size: 'sm', type: 'button', onclick: () => fileInput?.click() }}
            >
                Choose videos
            </Button>
            <input
                bind:this={fileInput}
                type="file"
                accept={['video/*', ...VIDEO_EXTENSIONS].join(',')}
                multiple
                class="hidden"
                data-testid="video-test-input"
                onchange={onFilesChosen}
            />
        </div>
        {#if files.length > 0}
            <ul class="mb-3 flex flex-col gap-1 text-sm" data-testid="video-test-files">
                {#each files as file, index (index)}
                    <li class="flex items-center justify-between gap-2">
                        <span class="min-w-0 truncate">{file.name}</span>
                        <span class="flex shrink-0 items-center gap-1 text-muted-foreground">
                            {formatBytes(file.size)}
                            <button
                                class="rounded p-0.5 hover:text-foreground"
                                aria-label={`Remove ${file.name}`}
                                onclick={() => removeFile(index)}
                            >
                                <X class="size-3.5" />
                            </button>
                        </span>
                    </li>
                {/each}
            </ul>
        {/if}
        <div class="mb-3 grid gap-1.5">
            <Label>Model</Label>
            {#if modelItems.length === 0}
                <p class="text-sm text-muted-foreground">The models folder has no models.</p>
            {:else}
                <Select
                    items={modelItems}
                    value={model}
                    onValueChange={(v) => (model = v)}
                    class="w-full"
                    testId="video-test-model"
                />
            {/if}
        </div>
        <div class="mb-4 flex items-center gap-2">
            <Button
                variant="default"
                icon={Play}
                buttonProps={{
                    size: 'sm',
                    disabled: !canRun,
                    onclick: () => model && start(files, model),
                    'data-testid': 'video-test-run'
                }}
            >
                {startMutation.isPending
                    ? `Uploading ${formatBytes(totalBytes)}…`
                    : `Run on ${files.length} video${files.length === 1 ? '' : 's'}`}
            </Button>
            {#if running}
                <span class="text-xs text-muted-foreground">A video test is running.</span>
            {/if}
        </div>
    {/if}

    <Typography variant="subtitle2" className="mb-2">Video tests</Typography>
    <div class="flex min-h-0 flex-1 flex-col overflow-y-auto dark:[color-scheme:dark]">
        {#if runsQuery.isLoading}
            <Spinner size="medium" align="center" />
        {:else if runsQuery.error}
            <Typography variant="body2" className="text-red-500"
                >Error loading video tests: {runsQuery.error.message}</Typography
            >
        {:else if runs.length === 0}
            <Typography variant="body2" className="text-muted-foreground"
                >No video test yet.</Typography
            >
        {:else}
            <ul class="flex flex-col gap-2">
                {#each runs as run (run.name)}
                    <VideoTestRunRow
                        {run}
                        onLog={() => (logName = run.name)}
                        onStop={() => stop(run.name)}
                    />
                {/each}
            </ul>
        {/if}
    </div>
</div>

{#if logRun}
    <VideoTestLogDialog
        name={logRun.name}
        active={isRunningTest(logRun)}
        onClose={() => (logName = null)}
    />
{/if}
