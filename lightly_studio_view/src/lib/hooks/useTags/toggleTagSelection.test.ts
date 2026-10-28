import { describe, expect, it, vi } from 'vitest';
import { toggleTagSelection } from './toggleTagSelection';

const defaultParams = {
    tagId: 't1',
    collectionId: 'col-1',
    currentSelected: new Set<string>()
};

describe('toggleTagSelection', () => {
    it('calls updateSelected with the tag added when it was not selected', () => {
        const updateSelected = vi.fn();

        toggleTagSelection({ ...defaultParams, updateSelected });

        expect(updateSelected).toHaveBeenCalledWith('col-1', new Set(['t1']));
    });

    it('calls updateSelected with the tag removed when it was already selected', () => {
        const updateSelected = vi.fn();

        toggleTagSelection({
            ...defaultParams,
            currentSelected: new Set(['t1', 't2']),
            updateSelected
        });

        expect(updateSelected).toHaveBeenCalledWith('col-1', new Set(['t2']));
    });
});
