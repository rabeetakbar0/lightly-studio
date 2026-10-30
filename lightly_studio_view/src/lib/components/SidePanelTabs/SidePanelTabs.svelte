<script lang="ts">
    import { cn } from '$lib/utils/shadcn';
    import { Boxes, ChartColumn, ChartNetwork, Dumbbell, Gauge, SearchCode } from '@lucide/svelte';
    import { Tooltip } from '$lib/components/ui/tooltip';
    import { useSidePanelTabs } from './useSidePanelTabs';

    interface Props {
        isImages: boolean;
        hasMediaWithEmbeddings: boolean;
        supportsEvaluation: boolean;
        supportsDistribution: boolean;
    }
    const { isImages, hasMediaWithEmbeddings, supportsEvaluation, supportsDistribution }: Props =
        $props();

    const { activePanel, toggle } = useSidePanelTabs();
</script>

<div class="flex w-14 flex-col gap-2 rounded-xl bg-card p-1.5">
    {#if hasMediaWithEmbeddings}
        <Tooltip
            content="Explore the embedding space"
            position="left"
            triggerClass="w-full"
            class="w-max"
        >
            <button
                class={cn(
                    'flex aspect-square w-full flex-col items-center justify-center gap-0.5 rounded-md p-1.5 text-[10px] font-medium transition-colors',
                    $activePanel === 'embeddingPlot'
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
                data-testid="side-panel-tabs-embed"
                aria-label="Embeddings"
                aria-pressed={$activePanel === 'embeddingPlot'}
                onclick={() => toggle($activePanel, 'embeddingPlot')}
            >
                <ChartNetwork class="size-4" />
                <span>Embed</span>
            </button>
        </Tooltip>
    {/if}
    {#if isImages}
        <Tooltip
            content="Write a query expression to filter your dataset"
            position="left"
            triggerClass="w-full"
            class="w-max"
        >
            <button
                class={cn(
                    'flex aspect-square w-full flex-col items-center justify-center gap-0.5 rounded-md p-1.5 text-[10px] font-medium transition-colors',
                    $activePanel === 'queryEditor'
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
                data-testid="side-panel-tabs-query"
                aria-label="Query"
                aria-pressed={$activePanel === 'queryEditor'}
                onclick={() => toggle($activePanel, 'queryEditor')}
            >
                <SearchCode class="size-4" />
                <span>Query</span>
            </button>
        </Tooltip>
    {/if}
    {#if supportsEvaluation}
        <Tooltip
            content="Review evaluation run results"
            position="left"
            triggerClass="w-full"
            class="w-max"
        >
            <button
                class={cn(
                    'flex aspect-square w-full flex-col items-center justify-center gap-0.5 rounded-md p-1.5 text-[10px] font-medium transition-colors',
                    $activePanel === 'evaluationRuns'
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
                data-testid="side-panel-tabs-eval"
                aria-label="Evaluation"
                aria-pressed={$activePanel === 'evaluationRuns'}
                onclick={() => toggle($activePanel, 'evaluationRuns')}
            >
                <Gauge class="size-4" />
                <span>Eval</span>
            </button>
        </Tooltip>
    {/if}
    {#if supportsDistribution}
        <Tooltip
            content="View dataset distribution"
            position="left"
            triggerClass="w-full"
            class="w-max"
        >
            <button
                class={cn(
                    'flex aspect-square w-full flex-col items-center justify-center gap-0.5 rounded-md p-1.5 text-[10px] font-medium transition-colors',
                    $activePanel === 'distribution'
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
                data-testid="side-panel-tabs-distribution"
                aria-label="Distribution"
                aria-pressed={$activePanel === 'distribution'}
                onclick={() => toggle($activePanel, 'distribution')}
            >
                <ChartColumn class="size-4" />
                <span>Distr</span>
            </button>
        </Tooltip>
    {/if}
    {#if isImages}
        <Tooltip
            content="Prepared datasets and training"
            position="left"
            triggerClass="w-full"
            class="w-max"
        >
            <button
                class={cn(
                    'flex aspect-square w-full flex-col items-center justify-center gap-0.5 rounded-md p-1.5 text-[10px] font-medium transition-colors',
                    $activePanel === 'train'
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
                data-testid="side-panel-tabs-train"
                aria-label="Train"
                aria-pressed={$activePanel === 'train'}
                onclick={() => toggle($activePanel, 'train')}
            >
                <Dumbbell class="size-4" />
                <span>Train</span>
            </button>
        </Tooltip>
        <Tooltip
            content="Detection models and their training history"
            position="left"
            triggerClass="w-full"
            class="w-max"
        >
            <button
                class={cn(
                    'flex aspect-square w-full flex-col items-center justify-center gap-0.5 rounded-md p-1.5 text-[10px] font-medium transition-colors',
                    $activePanel === 'models'
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
                data-testid="side-panel-tabs-models"
                aria-label="Models"
                aria-pressed={$activePanel === 'models'}
                onclick={() => toggle($activePanel, 'models')}
            >
                <Boxes class="size-4" />
                <span>Models</span>
            </button>
        </Tooltip>
    {/if}
</div>
