import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createMutation, useQueryClient } from '@tanstack/svelte-query';
import { useUpdateAnnotationsMutation } from './useUpdateAnnotationsMutation';

vi.mock('@tanstack/svelte-query', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@tanstack/svelte-query')>();
    return { ...actual, createMutation: vi.fn(), useQueryClient: vi.fn() };
});

const { invalidateAnnotationGridQueries, useInvalidateAnnotationGridQueries } = vi.hoisted(() => ({
    invalidateAnnotationGridQueries: vi.fn(),
    useInvalidateAnnotationGridQueries: vi.fn()
}));
vi.mock('$lib/hooks/useInvalidateAnnotationGridQueries', () => ({
    useInvalidateAnnotationGridQueries
}));

describe('useUpdateAnnotationsMutation', () => {
    const invalidateQueries = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
        useInvalidateAnnotationGridQueries.mockReturnValue(invalidateAnnotationGridQueries);
        vi.mocked(useQueryClient).mockReturnValue({
            invalidateQueries
        } as unknown as ReturnType<typeof useQueryClient>);
    });

    it('invalidates annotation-bearing grids and evaluation runs after a successful update', async () => {
        vi.mocked(createMutation).mockReturnValue({
            mutateAsync: vi.fn().mockResolvedValue(undefined)
        } as unknown as ReturnType<typeof createMutation>);

        const { updateAnnotations } = useUpdateAnnotationsMutation({
            getCollectionId: () => 'col-1'
        });
        await updateAnnotations([
            { annotation_id: 'ann-1', collection_id: 'col-1', label_name: 'dog' }
        ]);

        expect(useInvalidateAnnotationGridQueries).toHaveBeenCalledWith();
        expect(invalidateAnnotationGridQueries).toHaveBeenCalledWith('col-1');
        expect(invalidateQueries).toHaveBeenCalledWith({
            queryKey: [{ _id: 'getEvaluationRuns' }]
        });
    });
});
