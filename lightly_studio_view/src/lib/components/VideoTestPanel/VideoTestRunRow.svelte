<script lang="ts">
    import type { VideoTestRunView } from '$lib/api/lightly_studio_local';
    import { Button, Typography } from '$lib/components';
    import { Badge, type BadgeVariant } from '$lib/components/ui/badge';
    import { Copy, FileText, Square } from '@lucide/svelte';
    import { toast } from 'svelte-sonner';
    import { formatDate } from '$lib/utils';
    import { isRunningTest, progressLabel } from '$lib/hooks/useVideoTests/useVideoTests.svelte';

    interface Props {
        run: VideoTestRunView;
        onLog: () => void;
        onStop: () => void;
    }
    let { run, onLog, onStop }: Props = $props();

    const variants: Record<string, BadgeVariant> = {
        running: 'default',
        finished: 'secondary',
        failed: 'destructive',
        stopped: 'outline'
    };
    const active = $derived(isRunningTest(run));
    const percent = $derived(run.progress?.percent ?? 0);

    const seconds = (value: string | undefined) =>
        value === undefined || Number.isNaN(Number(value)) ? '–' : `${Number(value).toFixed(1)} s`;
    /** "arm detected · descent 9.9 s · ascent 5.2 s · blinks 0/5/6 (tip/middle/base)". */
    const summaryLabel = (summary: Record<string, string>): string => {
        if (Object.keys(summary).length === 0) return '';
        const detected = summary.GatearmDetected === 'True' ? 'arm detected' : 'arm NOT detected';
        const blinks = ['Tip', 'Middle', 'Base']
            .map((led) => summary[`${led}LightBlinkCount`] ?? '–')
            .join('/');
        return [
            detected,
            `descent ${seconds(summary.GatearmDescentTime)}`,
            `ascent ${seconds(summary.GatearmAscentTime)}`,
            `blinks ${blinks} (tip/middle/base)`
        ].join(' · ');
    };
    const copy = async (path: string) => {
        try {
            await navigator.clipboard.writeText(path);
            toast.success('Path copied');
        } catch {
            toast.error('Copying is not allowed here; select the path instead');
        }
    };
</script>

<li class="rounded-md border border-border bg-background p-3" data-testid="video-test-row">
    <div class="flex items-start justify-between gap-2">
        <div class="min-w-0 flex-1">
            <div class="flex items-center gap-2">
                <Typography variant="subtitle2" className="truncate">{run.name}</Typography>
                <Badge variant={variants[run.status] ?? 'outline'}>{run.status}</Badge>
            </div>
            <Typography variant="caption" className="block text-muted-foreground">
                {run.model} · {run.videos.length} video{run.videos.length === 1 ? '' : 's'} ·
                {formatDate(new Date(run.created_at))}
            </Typography>
            {#if active}
                <div
                    class="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-muted"
                    role="progressbar"
                    aria-valuenow={percent}
                    aria-valuemin={0}
                    aria-valuemax={100}
                >
                    <div class="h-full bg-primary transition-all" style:width={`${percent}%`}></div>
                </div>
                <p class="mt-1 text-xs text-muted-foreground" data-testid="video-test-progress">
                    {Math.round(percent)}% · {progressLabel(run.progress)}
                </p>
            {/if}
            {#if run.error}
                <Typography variant="caption" className="block text-red-500">{run.error}</Typography
                >
            {/if}
            <ul class="mt-2 flex flex-col gap-1.5">
                {#each run.videos as video (video.file_name)}
                    <li class="text-xs" data-testid="video-test-video">
                        {#if video.status === 'done' && video.output_path}
                            <div class="flex items-start gap-1">
                                <span class="min-w-0 break-all">
                                    Processed video saved at:
                                    <span class="font-mono">{video.output_path}</span>
                                </span>
                                <button
                                    class="shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground"
                                    aria-label={`Copy path of ${video.file_name}`}
                                    title="Copy path"
                                    onclick={() => copy(video.output_path ?? '')}
                                >
                                    <Copy class="size-3.5" />
                                </button>
                            </div>
                            {#if summaryLabel(video.summary)}
                                <span class="block text-muted-foreground"
                                    >{summaryLabel(video.summary)}</span
                                >
                            {/if}
                        {:else}
                            <span class="break-all">{video.file_name}</span>
                            <span
                                class={video.status === 'failed'
                                    ? 'text-red-500'
                                    : 'text-muted-foreground'}
                            >
                                · {video.status === 'failed'
                                    ? 'failed, see the log'
                                    : video.status === 'processing'
                                      ? 'processing'
                                      : active
                                        ? 'waiting'
                                        : 'not processed'}
                            </span>
                        {/if}
                    </li>
                {/each}
            </ul>
        </div>
        <div class="flex shrink-0 gap-1">
            <Button
                icon={FileText}
                ariaLabel={`Log of ${run.name}`}
                buttonProps={{ size: 'icon', title: 'Show log', onclick: onLog }}
            />
            {#if active}
                <Button
                    icon={Square}
                    ariaLabel={`Stop ${run.name}`}
                    buttonProps={{ size: 'icon', title: 'Stop the pipeline', onclick: onStop }}
                />
            {/if}
        </div>
    </div>
</li>
