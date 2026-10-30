<script lang="ts">
    import type { TrainConfig } from '$lib/api/lightly_studio_local';
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';
    import { Switch } from '$lib/components/ui/switch';
    import { Select } from '$lib/components/Select';
    import type { TrainField } from './trainFields';

    interface Props {
        fields: TrainField[];
        config: TrainConfig;
    }
    let { fields, config = $bindable() }: Props = $props();

    // Mutable view of the config, so one grid can set any key.
    const values = $derived(config as unknown as Record<string, unknown>);
    const set = (key: string, value: unknown) => {
        config = { ...config, [key]: value } as TrainConfig;
    };
    const parseNullable = (raw: string) => (raw.trim() === '' ? null : Number(raw));
</script>

<div class="grid gap-4 sm:grid-cols-2">
    {#each fields as field (field.key)}
        <div class="grid gap-1.5">
            {#if field.kind === 'boolean'}
                <div class="flex items-center gap-2 pt-1">
                    <Switch
                        id={`train-${field.key}`}
                        checked={Boolean(values[field.key])}
                        onCheckedChange={(checked: boolean) => set(field.key, checked)}
                        data-testid={`train-${field.key}`}
                    />
                    <Label for={`train-${field.key}`}>{field.label}</Label>
                </div>
            {:else if field.kind === 'select'}
                <Label>{field.label}</Label>
                <Select
                    items={(field.options ?? []).map((o) => ({ value: o, label: o }))}
                    value={String(values[field.key])}
                    onValueChange={(v) => set(field.key, v)}
                    class="w-full"
                    testId={`train-${field.key}`}
                />
            {:else if field.kind === 'nullable-number'}
                <Label for={`train-${field.key}`}>{field.label}</Label>
                <Input
                    id={`train-${field.key}`}
                    type="number"
                    step={field.step}
                    min={field.min}
                    value={values[field.key] ?? ''}
                    oninput={(e) => set(field.key, parseNullable(e.currentTarget.value))}
                    data-testid={`train-${field.key}`}
                />
            {:else}
                <Label for={`train-${field.key}`}>{field.label}</Label>
                <Input
                    id={`train-${field.key}`}
                    type="number"
                    step={field.step}
                    min={field.min}
                    required
                    value={values[field.key]}
                    oninput={(e) => set(field.key, Number(e.currentTarget.value))}
                    data-testid={`train-${field.key}`}
                />
            {/if}
            <p class="text-xs text-muted-foreground">{field.hint}</p>
        </div>
    {/each}
</div>
