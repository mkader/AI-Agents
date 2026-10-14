import { EventConfig, Handlers } from 'motia'
import { z } from 'zod'
import axios from 'axios'

export const config: EventConfig = {
  type: 'event',
  name: 'ScheduleTwitter',
  description: 'Schedules generated Twitter post using Typefully',
  subscribes: ['twitter-schedule'],
  emits: [],
  input: z.object({
    requestId: z.string(),
    url: z.string().url(),
    title: z.string(),
    content: z.object({
      thread: z.array(z.object({ content: z.string() }))
    })
  }),
  flows: ['content-generation']
}

export const handler: Handlers['ScheduleTwitter'] = async (input, { logger }) => {
  try {
    
    const apiKey = process.env.TYPEFULLY_API_KEY
    if (!apiKey) {
      throw new Error('TYPEFULLY_API_KEY is required for Twitter scheduling')
    }
    const socialSetId = process.env.TYPEFULLY_SOCIAL_SET_ID?.trim()
    if (!socialSetId) {
      throw new Error('TYPEFULLY_SOCIAL_SET_ID is required for Twitter scheduling')
    }
    const typefullyApiUrl = `https://api.typefully.com/v2/social-sets/${encodeURIComponent(socialSetId)}/drafts`
    const headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    }
    logger.info('Scheduling Twitter thread...')

    await axios.post(typefullyApiUrl, {
      platforms: {
        x: {
          enabled: true,
          posts: input.content.thread.map((tweet) => ({ text: tweet.content }))
        }
      }
    }, { headers })

    logger.info('Twitter draft created successfully. Visit Typefully to review and publish.')
  } catch (error) {
    logger.error(`Twitter thread scheduling failed: ${error instanceof Error ? error.message : String(error)}`)
    throw error
  }
}
