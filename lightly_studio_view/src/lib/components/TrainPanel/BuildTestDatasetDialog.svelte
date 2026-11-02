<script lang="ts">
    import type {
        BuildTestDatasetRequest,
        PipelineDatasetView
    } from '$lib/api/lightly_studio_local';
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Checkbox } from '$lib/components/ui/checkbox';
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';
    import { Select } from '$lib/components/Select';

    interface Props {
        datasets: PipelineDatasetView[];
        isPending: boolean;
        onConfirm: (body: BuildTestDatasetRequest) => void;
        onCancel: () => void;
    }
    let { datasets, isPending, onConfirm, onCancel }: Props = $props();

    let name = $state('');
    let sources = $state<string[]>([]);
    let mode = $state<'fraction' | 'count'>('fraction');
    let amount = $state(0.1);
    let strategy = $state<'embedding' | 'random'>('embedding');
    let groups = $state<number | null>(null);

    const pooled = $derived(
        datasets.filter((d) => sources.includes(d.name)).reduce((n, d) => n + d.images, 0)
    );
    const amountOk = $derived(
        mode === 'fraction' ? amount > 0 && amount < 1 : Number.isInteger(amount) && amount >= 1
    );
    const canSubmit = $derived(
        sources.length > 0 &&
            amountOk &&
            /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name.trim()) &&
            !isPending
    );
    const toggle = (dataset: string, on: boolean) => {
        sources = on ? [...sources, dataset] : sources.filter((s) => s !== dataset);
    };
    const submit = (event: SubmitEvent) => {
        event.preventDefault();
        if (!canSubmit) return;
        onConfirm({
            name: name.trim(),
            sources,
            fraction: mode === 'fraction' ? amount : null,
            count: mode === 'count' ? amount : null,
            split_strategy: strategy,
            groups: strategy === 'embedding' && groups ? groups : null
        });
    };
    const close = () => {
        if (!isPending) onCancel();
    };
</script>

<Dialog.Root bind:open={() => true, close}>
    <Dialog.Content>
        <form onsubmit={submit} novalidate>
            <Dialog.Header>
                <Dialog.Title>Build a test dataset</Dialog.Title>
                <Dialog.Description>
                    Draws images from every split of the chosen datasets (each file name once) into
                    a new dataset that holds only a test split. Spread takes the sample from every
                    similarity group. The sources stay unchanged.
                </Dialog.Description>
            </Dialog.Header>
            <div class="grid gap-3 py-4">
                <div class="grid gap-1.5">
                    <Label>Draw from</Label>
                    <ul
                        class="grid max-h-40 gap-1 overflow-y-auto rounded-md border border-border p-2"
                    >
                        {#each datasets as dataset (dataset.name)}
                            <li class="flex items-center gap-2 text-sm">
                                <Checkbox
                                    checked={sources.includes(dataset.name)}
                                    onCheckedChange={(v: boolean) => toggle(dataset.name, v)}
                                    aria-label={`Draw from ${dataset.name}`}
                                />
                                <span class="truncate">{dataset.name}</span>
                                <span class="text-muted-foreground">({dataset.images} images)</span>
                            </li>
                        {/each}
                    </ul>
                    <p class="text-xs text-muted-foreground">{pooled} images in the pool.</p>
                </div>
                <div class="grid gap-3 sm:grid-cols-2">
                    <div class="grid gap-1.5">
                        <Label>Take</Label>
                        <Select
                            items={[
                                { value: 'fraction', label: 'A share of the pool' },
                                { value: 'count', label: 'A number of images' }
                            ]}
                            value={mode}
                            onValueChange={(v) => {
                                mode = v as 'fraction' | 'count';
                                amount = mode === 'fraction' ? 0.1 : 100;
                            }}
                            class="w-full"
                            testId="build-test-mode"
                        />
                    </div>
                    <div class="grid gap-1.5">
                        <Label for="build-test-amount"
                            >{mode === 'fraction' ? 'Share' : 'Images'}</Label
                        >
                        <Input
                            id="build-test-amount"
                            type="number"
                            min={mode === 'fraction' ? 0.01 : 1}
                            max={mode === 'fraction' ? 0.99 : undefined}
                            step={mode === 'fraction' ? 0.05 : 1}
                            bind:value={amount}
                            data-testid="build-test-amount"
                        />
                    </div>
                    <div class="grid gap-1.5">
                        <Label>Spread</Label>
                        <Select
                            items={[
                                { value: 'embedding', label: 'Over similarity groups' },
                                { value: 'random', label: 'Random' }
                            ]}
                            value={strategy}
                            onValueChange={(v) => (strategy = v as 'embedding' | 'random')}
                            class="w-full"
                            testId="build-test-strategy"
                        />
                    </div>
                    <div class="grid gap-1.5">
                        <Label for="build-test-groups">Groups</Label>
                        <Input
                            id="build-test-groups"
                            type="number"
                            min={2}
                            step={1}
                            placeholder="auto"
                            disabled={strategy !== 'embedding'}
                            value={groups ?? ''}
                            oninput={(e) =>
                                (groups = e.currentTarget.value
                                    ? Number(e.currentTarget.value)
                                    : null)}
                        />
                    </div>
                </div>
                <div class="grid gap-1.5">
                    <Label for="build-test-name">Test dataset name</Label>
                    <Input
                        id="build-test-name"
                        bind:value={name}
                        placeholder="e.g. gate_test_v1"
                        data-testid="build-test-name"
                    />
                </div>
            </div>
            <Dialog.Footer>
                <Button
                    variant="outline"
                    buttonProps={{ type: 'button', disabled: isPending, onclick: close }}
                    >Cancel</Button
                >
                <Button
                    {isPending}
                    buttonProps={{
                        type: 'submit',
                        disabled: !canSubmit,
                        'data-testid': 'build-test-confirm'
                    }}
                >
                    {isPending ? 'Building…' : 'Build test dataset'}
                </Button>
            </Dialog.Footer>
        </form>
    </Dialog.Content>
</Dialog.Root>
