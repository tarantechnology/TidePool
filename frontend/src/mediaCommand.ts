/** Conservative local interpretation of commands that affect the exported media. */
export type MediaCommand =
  | {kind:'trim'; start:number; end:number; exportFormat:'mp4'|'wav'|'mp3'|null; download:boolean}
  | {kind:'export'; format:'mp4'|'wav'|'mp3'|null}
  | {kind:'invalid'; message:string}

const words: Record<string,number> = {zero:0,one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,eleven:11,twelve:12,thirteen:13,fourteen:14,fifteen:15,twenty:20,thirty:30,forty:40,'forty-five':45,sixty:60}
const time = '(?:\\d+:\\d{1,2}(?:\\.\\d+)?|\\d+(?:\\.\\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty-five|forty|sixty)'
const unit = '(?:\\s*(?:seconds?|secs?|s))?'
function seconds(value:string):number | null {
  if (value.includes(':')) {
    const [minutes,secs]=value.split(':').map(Number)
    return secs < 60 ? minutes*60+secs : null
  }
  return words[value] ?? (Number.isFinite(Number(value)) ? Number(value) : null)
}

export function mediaCommand(input:string):MediaCommand|null {
  const t=input.toLowerCase().trim().replace(/[.!?]+$/,'').replace(/^(?:please |can you |could you |would you )+/,'')
  const range=new RegExp(`^(?:trim|crop|keep)(?: (?:the|this|my))?(?: (?:video|clip|audio|file))?(?: (?:down )?to| from| between)?\\s+(${time})${unit}\\s*(?:to|through|until|and|[-–])\\s*(${time})${unit}(?:\\s+(?:and |then )?(?:export|download|save)(?: (?:it|the (?:video|clip|audio|file)))?(?: (?:as|to) (?:an? )?(mp4|wav|mp3))?)?$`,'i').exec(t)
  if (range) {
    const start=seconds(range[1]),end=seconds(range[2])
    if (start === null || end === null || start >= end) return {kind:'invalid',message:'Choose a valid range with the end after the start.'}
    return {kind:'trim',start,end,exportFormat:(range[3] as 'mp4'|'wav'|'mp3'|undefined) ?? null,download:/\b(?:export|download|save)\b/.test(t)}
  }
  const exp=/^(?:export|download|save)(?: (?:it|this|the|my))?(?: (?:video|clip|audio|file|project))?(?: (?:as|to) (?:an? )?(mp4|wav|mp3))?$/.exec(t)
  if (exp) return {kind:'export',format:(exp[1] as 'mp4'|'wav'|'mp3'|undefined) ?? null}
  if (/^(?:trim|crop|cut)(?:\s|$)/.test(t)) return {kind:'invalid',message:'Try “trim from 3 seconds to 4 seconds”. This keeps that range.'}
  return null
}
