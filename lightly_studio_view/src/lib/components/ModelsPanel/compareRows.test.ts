import { describe, expect, it } from 'vitest';
import type { PipelineModelView } from '$lib/api/lightly_studio_local';
import { compareRows } from './compareRows';

const model = (name: string, tests: object[]): PipelineModelView =>
    ({
        name,
        classes: ['GateArm'],
        onnx_path: '',
        has_checkpoint: true,
        size_bytes: 1,
        modified_at: '',
        depth: 0.67,
        width: 0.75,
        card: {
            run_name: name,
            dataset: {
                name: 'd',
                classes: ['GateArm'],
                images: 1,
                boxes: 1,
                splits: {},
                annotation_source: '',
                sources: []
            },
            baseline: null,
            config: {},
            best_ap50_95: null,
            evaluations: [],
            trained_epochs: 1,
            created_at: '',
            finished_at: '',
            trained: true,
            tests
        }
    }) as unknown as PipelineModelView;

describe('compareRows', () => {
    it('adds one block of rows per test dataset with the latest result of each model', () => {
        const a = model('a', [
            {
                dataset: 't1',
                ap50_95: 0.5,
                ap50: 0.8,
                per_class_ap: { GateArm: 0.5 },
                overlap_images: 0
            },
            {
                dataset: 't1',
                ap50_95: 0.7,
                ap50: 0.9,
                per_class_ap: { GateArm: 0.7 },
                overlap_images: 2
            }
        ]);
        const b = model('b', []);
        const rows = compareRows([a, b]).filter((r) => r.group === 'Test');
        expect(rows.map((r) => [r.label, ...r.values, r.differs])).toEqual([
            ['t1 AP50:95', '70.0', '—', true],
            ['t1 AP50', '90.0', '—', true],
            ['t1 AP GateArm', '70.0', '—', true],
            ['t1 seen in training', '2', '—', true]
        ]);
    });
});
