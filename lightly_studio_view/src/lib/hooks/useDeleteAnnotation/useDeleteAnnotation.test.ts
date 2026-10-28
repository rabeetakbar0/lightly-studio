import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createMutation, useQueryClient } from '@tanstack/svelte-query';
import { useDeleteAnnotation } from './useDeleteAnnotation';

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

describe('useDeleteAnnotation', () => {
    const invalidateQueries = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
        useInvalidateAnnotationGridQueries.mockReturnValue(invalidateAnnotationGridQueries);
        vi.mocked(useQueryClient).mockReturnValue({
            invalidateQueries
        } as unknown as ReturnType<typeof useQueryClient>);
    });

    it('invalidates annotation-bearing grids and evaluation runs after a successful delete', async () => {
        vi.mocked(createMutation).mockReturnValue({
            mutate: (_vars: unknown, opts: { onSuccess: () => void }) => {
                opts.onSuccess();
            }
        } as unknown as ReturnType<typeof createMutation>);

        const { deleteAnnotation } = useDeleteAnnotation({ getCollectionId: () => 'col-1' });
        await deleteAnnotation('ann-1');

        expect(useInvalidateAnnotationGridQueries).toHaveBeenCalledWith();
        expect(invalidateAnnotationGridQueries).toHaveBeenCalledWith('col-1');
        expect(invalidateQueries).toHaveBeenCalledWith({
            queryKey: [{ _id: 'getEvaluationRuns' }]
        });
    });

    it('invalidates the explicitly passed collection instead of the current one', async () => {
        vi.mocked(createMutation).mockReturnValue({
            mutate: (_vars: unknown, opts: { onSuccess: () => void }) => {
                opts.onSuccess();
            }
        } as unknown as ReturnType<typeof createMutation>);

        const { deleteAnnotation } = useDeleteAnnotation({ getCollectionId: () => 'new-col' });
        await deleteAnnotation('ann-1', 'col-1');

        expect(invalidateAnnotationGridQueries).toHaveBeenCalledWith('col-1');
    });
});
