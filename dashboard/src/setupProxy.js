/**
 * Copyright 2023 Cisco Systems, Inc. and its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'https://flame-blue-b.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/api' : '/',
        },
    })
  );

  app.use(
    '/design/api',
    createProxyMiddleware({
      target: 'https://flame-blue-b.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/design/api' : '/',
        },
    })
  )

  app.use(
    '/jobs/api',
    createProxyMiddleware({
      target: 'https://flame-blue-b.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/jobs/api' : '/',
        },
    })
  )

  app.use(
    '/jobs/mlflow/get-artifact',
    createProxyMiddleware({
      target: 'https://flame-mlflow.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/jobs/mlflow/get-artifact' : '/get-artifact',
        },
    })
  )

  app.use(
    '/mlflow',
    createProxyMiddleware({
      target: 'https://flame-mlflow.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/mlflow' : '/ajax-api/2.0/',
        },
    })
  )

  app.use(
    '/jobs/mlflow',
    createProxyMiddleware({
      target: 'https://flame-mlflow.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/jobs/mlflow' : '/ajax-api/2.0/',
        },
    })
  )
};