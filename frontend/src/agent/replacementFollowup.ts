/** Carry a requested example time into the previous replacement request. */
export function replacementFollowup(previous:string,text:string):string|null {
  const match=text.trim().match(/^(?:(?:at|around|near)\s+)?(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)?[.!?]?$/i)
  return previous && match ? `${previous.replace(/[.!?]+$/,'')} at ${match[1]} seconds` : null
}
