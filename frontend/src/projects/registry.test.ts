import { beforeEach, describe, expect, it } from 'vitest'
import { claimShareLink, fileBase, forgetProject, getProject, listProjects, loadSteps, nameFromFile, parseShareLink, renameProject, saveProject, saveSteps, shareLink } from './registry'

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

describe('saving and sharing a project', () => {
  beforeEach(() => localStorage.clear())
  it('keeps a name the user gave it', () => {
    saveProject({ id: 'a', token: 't', name: 'rain scene' })
    renameProject('a', '  Batman rain, final  ')
    expect(getProject('a')).toMatchObject({ name: 'Batman rain, final', named: true })
    renameProject('a', '   ')
    expect(getProject('a')?.name).toBe('Batman rain, final')
  })
  it('round-trips a project link, token in the # part only', () => {
    const link = shareLink({ id: 'p 1', token: 'tok/+=', name: 'Rain & thunder' }, 'https://tide.example/')
    expect(link.startsWith('https://tide.example/#/app/open/')).toBe(true)
    const hash = link.slice(link.indexOf('#'))
    expect(parseShareLink(hash)).toEqual({ id: 'p 1', token: 'tok/+=', name: 'Rain & thunder' })
    expect(claimShareLink(hash)).toBe('#/app/p/p%201')
    expect(getProject('p 1')).toMatchObject({ token: 'tok/+=', name: 'Rain & thunder', named: true })
  })
  it('ignores other addresses and links without a token', () => {
    expect(claimShareLink('#/app/p/x')).toBeNull()
    expect(parseShareLink('#/app/open/x?n=Hi')).toBeNull()
    expect(listProjects()).toEqual([])
  })
  it('turns a name into a safe file name', () => {
    expect(fileBase('Rain: final/v2?')).toBe('Rain final v2')
    expect(fileBase('')).toBe('tidepool')
    expect(fileBase(null)).toBe('tidepool')
  })
})
