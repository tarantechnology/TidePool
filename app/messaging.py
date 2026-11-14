"""Messaging boundary. The local provider persists an authenticated project's outbox."""
from typing import Protocol
from .core import connect, uid

class MessagingProvider(Protocol):
    def send(self, project_id: str, text: str) -> None: ...

class LocalMessaging:
    def send(self, project_id: str, text: str) -> None:
        with connect() as db:
            db.execute('INSERT INTO messages(id,project_id,text) VALUES(?,?,?)',(uid(),project_id,text))

class PhotonMessaging:
    def send(self, project_id: str, text: str) -> None:
        raise NotImplementedError('Select a Photon product and verify its API and credentials before enabling this adapter')

provider = LocalMessaging()
