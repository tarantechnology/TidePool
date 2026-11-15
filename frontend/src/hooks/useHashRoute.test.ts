import { describe, expect, it } from 'vitest'
import { parseHash } from './useHashRoute'

describe('parseHash', () => {
  it('reads the landing page, projects, a new project and an open one', () => {
    expect(parseHash('')).toEqual({ route: 'landing', projectId: null })
    expect(parseHash('#/')).toEqual({ route: 'landing', projectId: null })
    expect(parseHash('#/app')).toEqual({ route: 'projects', projectId: null })
    expect(parseHash('#/app/')).toEqual({ route: 'projects', projectId: null })
    expect(parseHash('#/app/new')).toEqual({ route: 'editor', projectId: null })
    expect(parseHash('#/app/p/abc123')).toEqual({ route: 'editor', projectId: 'abc123' })
    expect(parseHash('#/app/p/')).toEqual({ route: 'projects', projectId: null })
  })
})
