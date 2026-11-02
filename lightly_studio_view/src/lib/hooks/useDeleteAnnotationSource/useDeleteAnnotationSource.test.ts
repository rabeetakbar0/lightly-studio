import { describe, expect, it } from 'vitest';
import { deletedSummary } from './useDeleteAnnotationSource.svelte';

describe('deletedSummary', () => {
    it('lists only what was deleted', () => {
        const base = { deleted_sources: 1, deleted_evaluations: 0, deleted_classes: 0 };
        expect(deletedSummary({ ...base, deleted_annotations: 1 })).toBe('1 annotation');
        expect(
            deletedSummary({
                ...base,
                deleted_annotations: 120,
                deleted_evaluations: 1,
                deleted_classes: 2
            })
        ).toBe('120 annotations, 1 evaluation and 2 unused classes');
    });
});
