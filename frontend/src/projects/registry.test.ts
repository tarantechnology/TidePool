import { beforeEach, describe, expect, it } from 'vitest'
import { forgetProject, getProject, listProjects, loadSteps, nameFromFile, saveProject, saveSteps } from './registry'

// The tests run in node, which has no localStorage.
const store = new Map<string, string>()
Object.assign(globalThis, { localStorage: {
  getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v),
  removeItem: (k: string) => void store.delete(k), clear: () => store.clear(),
} })

describe('project registry', () => {
  beforeEach(() => localStorage.clear())
  it('remembers projects, newest opened first', () => {
    saveProject({ id: 'a', token: 't1', name: 'Rain' })
    saveProject({ id: 'b', token: 't2', name: 'Song' })
    expect(listProjects().map((p) => p.id)).toEqual(['b', 'a'])
    saveProject({ id: 'a' }, true)
    expect(listProjects()[0].id).toBe('a')
    expect(getProject('a')?.token).toBe('t1')
  })
  it('needs a token to add a project, not to update one', () => {
    saveProject({ id: 'x', name: 'No token' })
    expect(getProject('x')).toBeNull()
    saveProject({ id: 'x', token: 't', name: 'Clip' })
    saveProject({ id: 'x', layers: 4 })
    expect(getProject('x')).toMatchObject({ token: 't', name: 'Clip', layers: 4 })
  })
  it('forgets a project and its undo steps', () => {
    saveProject({ id: 'a', token: 't', name: 'Rain' })
    saveSteps('a', { past: [{ label: 'Explode', before: '1', after: '2' }], future: [] })
    expect(loadSteps('a').past).toHaveLength(1)
    forgetProject('a')
    expect(getProject('a')).toBeNull()
    expect(loadSteps('a').past).toHaveLength(0)
  })
  it('names a project from its file', () => {
    expect(nameFromFile('batman_rain-scene.mp4')).toBe('batman rain scene')
    expect(nameFromFile('')).toBe('Untitled clip')
  })
})
