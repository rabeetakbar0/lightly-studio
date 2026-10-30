import type { PipelineModelView } from '$lib/api/lightly_studio_local';
import { TRAIN_FIELD_GROUPS } from '$lib/components/TrainPanel/trainFields';
import { baselineLabel } from '$lib/hooks/usePipelineModels/usePipelineModels.svelte';

export interface CompareRow {
    group: string;
    label: string;
    /** One cell per model, in the order given; "—" when the model has no history. */
    values: string[];
    /** True when at least two models have different values. */
    differs: boolean;
}

const cell = (value: unknown): string => {
    if (value === null || value === undefined || value === '') return '—';
    if (typeof value === 'boolean') return value ? 'yes' : 'no';
    if (Array.isArray(value)) return value.join(', ');
    return String(value);
};

const row = (group: string, label: string, values: string[]): CompareRow => ({
    group,
    label,
    values,
    differs: new Set(values).size > 1
});

/** The side-by-side table of the models: dataset, run results, then every setting. */
export function compareRows(models: PipelineModelView[]): CompareRow[] {
    const cards = models.map((m) => m.card ?? null);
    const rows: CompareRow[] = [
        row(
            'Model',
            'Classes',
            models.map((m) => cell(m.classes))
        ),
        row(
            'Model',
            'Checkpoint',
            models.map((m) => (m.has_checkpoint ? 'yes' : 'no'))
        ),
        row(
            'Dataset',
            'Name',
            cards.map((c) => cell(c?.dataset.name))
        ),
        row(
            'Dataset',
            'Images',
            cards.map((c) => cell(c?.dataset.images))
        ),
        row(
            'Dataset',
            'Boxes',
            cards.map((c) => cell(c?.dataset.boxes))
        ),
        row(
            'Dataset',
            'Annotation source',
            cards.map((c) => cell(c?.dataset.annotation_source))
        ),
        row(
            'Dataset',
            'Merged from',
            cards.map((c) => cell(c?.dataset.sources))
        ),
        row(
            'Result',
            'Baseline',
            cards.map((c) => (c ? baselineLabel(c.baseline) : '—'))
        ),
        row(
            'Result',
            'Epochs trained',
            cards.map((c) => cell(c?.trained_epochs))
        ),
        row(
            'Result',
            'Best AP50:95',
            cards.map((c) =>
                c?.best_ap50_95 == null ? '—' : `${(c.best_ap50_95 * 100).toFixed(1)}`
            )
        ),
        row(
            'Result',
            'Trained',
            cards.map((c) => cell(c?.finished_at?.slice(0, 10)))
        )
    ];
    for (const group of TRAIN_FIELD_GROUPS) {
        for (const field of group.fields) {
            rows.push(
                row(
                    group.label,
                    field.label,
                    cards.map((c) => cell(c?.config[field.key]))
                )
            );
        }
    }
    return rows;
}
