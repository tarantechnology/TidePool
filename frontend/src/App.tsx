import { useEffect } from 'react'
import Editor from './Editor'
import { EntryRipple } from './components/EntryRipple'
import { useHashRoute } from './hooks/useHashRoute'
import Landing from './pages/Landing'
import Projects from './pages/Projects'

export default function App() {
  const { route, projectId, visit } = useHashRoute()
  useEffect(() => {
    if (route !== 'landing') window.scrollTo(0, 0)
  }, [route, visit])
  return (
    <>
      {/* A fresh editor for every project opened, so nothing from the last one carries over. */}
      {route === 'editor' ? <Editor key={visit} projectId={projectId} /> : route === 'projects' ? <Projects /> : <Landing />}
      {/* The site's entrance: plays once when you arrive, whichever page you arrive on. */}
      <EntryRipple />
    </>
  )
}
