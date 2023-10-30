const CustomTick = (data: any) => {
  if (typeof data.payload.value === 'string') {
    const lines = data.payload.value.split(' ');

    return (
      <g transform={`translate(${data.x},${data.y})`}>
          <text fontSize="10px" x={0} y={0} dy={16} fill="#666">
            <tspan textAnchor="middle" x="0">
              {lines[0]}
            </tspan>
            <tspan textAnchor="middle" x="0" dy="10">
              {lines[1]}
            </tspan>
          </text>
        </g>
    )
  }

  return (
    <g transform={`translate(${data.x},${data.y})`}>
        <text fontSize="10px" x={0} y={0} dy={16} fill="#666">
          <tspan textAnchor="middle" x="0">
            {data.payload.value}
          </tspan>
        </text>
      </g>
  )
}

export default CustomTick