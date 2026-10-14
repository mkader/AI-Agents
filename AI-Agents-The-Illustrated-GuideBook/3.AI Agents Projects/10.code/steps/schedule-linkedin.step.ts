import { EventConfig, Handlers } from 'motia'
import { z } from 'zod'
import axios from 'axios'

export const config: EventConfig = {
  type: 'event',
  name: 'ScheduleLinkedin',
  description: 'Schedules generated LinkedIn post using Typefully',
  subscribes: ['linkedin-schedule'],
  emits: [],
  input: z.object({
    requestId: z.string(),
    url: z.string().url(),
    title: z.string(),
    content: z.object({ post: z.string() })
  }),
  flows: ['content-generation']
}

export const handler: Handlers['ScheduleLinkedin'] = async (input, { logger }) => {
  try {
    const apiKey = process.env.TYPEFULLY_API_KEY
    if (!apiKey) {
      throw new Error('TYPEFULLY_API_KEY is required for LinkedIn scheduling')
    }
    const socialSetId = process.env.TYPEFULLY_SOCIAL_SET_ID?.trim()
    if (!socialSetId) {
      throw new Error('TYPEFULLY_SOCIAL_SET_ID is required for LinkedIn scheduling')
    }
    const typefullyApiUrl = `https://api.typefully.com/v2/social-sets/${encodeURIComponent(socialSetId)}/drafts`
    const headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    }

    logger.info('Scheduling LinkedIn post...')

    await axios.post(typefullyApiUrl, {
      platforms: {
        linkedin: {
          enabled: true,
          posts: [{ text: input.content.post }]
        }
      }
    }, { headers })

    logger.info('LinkedIn draft created successfully. Visit Typefully to review and publish.')
  } catch (error) {
    logger.error(`LinkedIn post scheduling failed: ${error instanceof Error ? error.message : String(error)}`)
    throw error
  }
}
