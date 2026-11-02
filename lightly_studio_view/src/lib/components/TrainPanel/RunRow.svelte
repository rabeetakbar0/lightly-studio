<script lang="ts">
    import type { TrainingRunView } from '$lib/api/lightly_studio_local';
    import { Button, Typography } from '$lib/components';
    import { Badge, type BadgeVariant } from '$lib/components/ui/badge';
    import { FileText, FlaskConical, Square, Trash2 } from '@lucide/svelte';
    import { formatDate } from '$lib/utils';
    import {
        isActiveRun,
        testResultLabel
    } from '$lib/hooks/usePipelineTraining/usePipelineTraining.svelte';

    interface Props {
        run: TrainingRunView;
        onLog: () => void;
        onStop: () => void;
        onDelete: () => void;
        onTest?: () => void;
    }
    let { run, onLog, onStop, onDelete, onTest }: Props = $props();

    const variants: Record<string, BadgeVariant> = {
        running: 'default',
        exporting: 'default',
        finished: 'secondary',
        failed: 'destructive',
        stopped: 'outline'
    };
    const active = $derived(isActiveRun(run));
    const isTest = $derived(run.kind === 'test');
    const canTest = $derived(
        !!onTest && run.kind === 'train' && run.status === 'finished' && !!run.onnx_path
    );
    const progress = $derived.by(() => {
        if (isTest) return run.test_result ? testResultLabel(run.test_result) : '';
        const p = run.progress;
        const parts: string[] = [];
        if (p.max_epoch > 0) parts.push(`epoch ${p.epoch}/${p.max_epoch}`);
        if (active && p.max_iteration > 0) parts.push(`iter ${p.iteration}/${p.max_iteration}`);
        if (active && p.eta) parts.push(`ETA ${p.eta}`);
        if (p.best_ap50_95 !== null) parts.push(`best AP ${(p.best_ap50_95 * 100).toFixed(1)}`);
        return parts.join(' · ');
    });
</script>

<li class="rounded-md border border-border bg-background p-3" data-testid="run-row">
    <div class="flex items-start justify-between gap-2">
        <div class="min-w-0">
            <div class="flex items-center gap-2">
                <Typography variant="subtitle2" className="truncate"
                    >{isTest ? `${run.model} on ${run.dataset}` : run.name}</Typography
                >
                {#if isTest}<Badge variant="outline">test</Badge>{/if}
                <Badge variant={variants[run.status] ?? 'outline'}>{run.status}</Badge>
            </div>
            <Typography variant="caption" className="block text-muted-foreground">
                {isTest ? 'test split of' : 'on'}
                {run.dataset} · {run.classes.join(', ')} · {formatDate(new Date(run.created_at))}
            </Typography>
            {#if progress}
                <Typography variant="caption" className="block text-muted-foreground"
                    >{progress}</Typography
                >
            {/if}
            {#if run.error}
                <Typography variant="caption" className="block text-red-500">{run.error}</Typography
                >
            {/if}
            {#if run.onnx_path}
                <Typography variant="caption" className="block text-muted-foreground">
                    Exported: {run.onnx_path}
                </Typography>
            {/if}
        </div>
        <div class="flex shrink-0 gap-1">
            {#if canTest}
                <Button
                    icon={FlaskConical}
                    ariaLabel={`Test ${run.name}`}
                    buttonProps={{ size: 'icon', title: 'Test on a test dataset', onclick: onTest }}
                />
            {/if}
            <Button
                icon={FileText}
                ariaLabel={`Log of ${run.name}`}
                buttonProps={{ size: 'icon', title: 'Show log', onclick: onLog }}
            />
            {#if active}
                <Button
                    icon={Square}
                    ariaLabel={`Stop ${run.name}`}
                    buttonProps={{ size: 'icon', title: 'Stop training', onclick: onStop }}
                />
            {:else}
                <Button
                    icon={Trash2}
                    ariaLabel={`Delete ${run.name}`}
                    buttonProps={{ size: 'icon', title: 'Delete run folder', onclick: onDelete }}
                />
            {/if}
        </div>
    </div>
</li>
