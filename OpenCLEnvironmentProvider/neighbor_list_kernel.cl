
__kernel void neighbor_list_cl(__global float3 *scaled_positions, __global float3 *offset, __global int *neighborhood_idx,
	uint n_atoms, uint max_nbh, float cutoff,
	float c11, float c12, float c13,
	float c21, float c22, float c23,
	float c31, float c32, float c33)
{
	uint i = get_global_id(0);
	uint k = max_nbh*i;

	float R_sq;
	float cutoff_sq = cutoff*cutoff - 0.000001f;

	float3 local_position = scaled_positions[i];

	float3 ds = (float3)(0.0f);
	float3 o  = (float3)(0.0f);

	float3 a = (float3)(c11, c12, c13);
	float3 b = (float3)(c21, c22, c23);
	float3 c = (float3)(c31, c32, c33);

	float3 denominator = dot(a, cross(b, c));
	int n1 = round(cutoff*length(cross(b, c)/denominator));
	int n2 = round(cutoff*length(cross(c, a)/denominator));
	int n3 = round(cutoff*length(cross(a, b)/denominator));

	for(int r1 = -n1; r1 < n1+1; r1++)
	{
	for(int r2 = -n2; r2 < n2+1; r2++)
	{
	for(int r3 = -n3; r3 < n3+1; r3++)
	{
		float3 shift = (float3)(r1, r2, r3);

		for (uint j = 0u; j < n_atoms; j++)
		{
			if (i == j && r1 == 0 && r2 == 0 && r3 == 0) continue;

			ds = scaled_positions[j] - local_position;
			o  = -round(ds) + shift;
			ds = ds + o;

			ds = ds.x*a + ds.y*b + ds.z*c;
			R_sq = ds.x*ds.x + ds.y*ds.y + ds.z*ds.z;

			if (R_sq < cutoff_sq)
			{
				neighborhood_idx[k] = j;
				offset[k] = o;
				k++;
			}
		}
	}
	}
	}
	if (k > max_nbh)
	{
		printf("\nWARNING! found %i neighbours for atom %i which larger than max_nbh=%i.\n         try and increase number_density", k, i, max_nbh);
	}
}
